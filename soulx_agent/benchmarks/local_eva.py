"""EVA caller provider using local ASR, a local LLM, and IndexTTS."""

import asyncio
import contextlib
from typing import Annotated, Literal

from pydantic import BaseModel, Field


def install():
    import eva.models.config as config
    import eva.user_simulator.factory as factory
    from eva.user_simulator.audio_bridge import BotToBotAudioBridge
    from eva.user_simulator.base import AbstractUserSimulator

    from soulx_agent.audio import SoulXAudio
    from soulx_agent.local_caller import LocalCaller, synthesize_caller

    try:
        import audioop
    except ImportError:
        import audioop_lts as audioop

    class LocalSimulatorConfig(BaseModel):
        provider: Literal["local"] = "local"

    class LocalRunConfig(config.RunConfig):
        user_simulator: Annotated[
            config.ElevenLabsSimulatorConfig
            | config.OpenAIRealtimeSimulatorConfig
            | LocalSimulatorConfig,
            Field(discriminator="provider"),
        ] = Field(default_factory=LocalSimulatorConfig)

    class LocalUserSimulator(AbstractUserSimulator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, provider="local")
            if self._perturbation_config and self._perturbation_config.accent is not None:
                raise ValueError("Local caller does not emulate ElevenLabs accent variants")
            self.audio_queue = asyncio.Queue(maxsize=128)
            self.heard = asyncio.Queue(maxsize=32)

        async def listen(self, recognizer):
            while True:
                mulaw = await self.audio_queue.get()
                state = await recognizer.process(audioop.ulaw2lin(mulaw, 2))
                if state["state"] == "speak" and state.get("text", "").strip():
                    text = state["text"].strip()
                    self._on_assistant_speaks(text)
                    await self.heard.put(text)

        async def speak(self, text):
            if not text:
                return
            self._on_user_speaks(text)
            self._audio_interface.output(await synthesize_caller(text))
            self._audio_interface.notify_user_utterance_complete()
            # The official bridge paces audio at real time; do not overlap the next caller turn.
            while self._audio_interface.is_caller_playing():
                await asyncio.sleep(0.02)

        async def converse(self):
            caller = LocalCaller(self._build_prompt())
            opening = self.goal["starting_utterance"]
            caller.messages.append({"role": "assistant", "content": opening})
            await self.speak(opening)
            while not self._conversation_done.is_set():
                text = await self.heard.get()
                response = await caller.reply(text)
                await self.speak(response["utterance"])
                if response["end_call"]:
                    self._on_conversation_end("goodbye")

        async def run_conversation(self):
            recognizer = SoulXAudio(input_rate=8000)
            tasks = []
            self._audio_interface = BotToBotAudioBridge(
                websocket_uri=self.server_url,
                conversation_id=self.output_dir.name,
                record_callback=self._record_audio,
                event_logger=self.event_logger,
                conversation_done_callback=self._on_conversation_end,
                perturbator=self._perturbator,
                disconnect_reason="assistant_disconnect",
            )
            try:
                await recognizer.connect()
                await self._audio_interface.start_async()
                self._audio_interface.start(self.audio_queue.put_nowait)
                self.event_logger.log_connection_state(
                    "connected",
                    {
                        "caller_provider": "local",
                        "perception": "SoulX ASR",
                        "synthesis": "IndexTTS",
                    },
                )
                tasks = [
                    asyncio.create_task(self.listen(recognizer)),
                    asyncio.create_task(self.converse()),
                    asyncio.create_task(self._conversation_done.wait()),
                ]
                done, _ = await asyncio.wait(
                    tasks, timeout=self.timeout, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    self._on_conversation_end("time_limit_exceeded")
                for task in done:
                    task.result()
            except Exception as error:
                self.event_logger.log_error(str(error))
                self._on_conversation_end("error")
                raise
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.signal_conversation_ending(self._end_reason)
                with contextlib.suppress(Exception):
                    await self._audio_interface.stop_async()
                await recognizer.close()
                self._save_clean_user_audio(16000)
                self.event_logger.log_connection_state(
                    "session_ended", {"reason": self._end_reason}
                )
                self.event_logger.save()
            return self._end_reason

    original_factory = factory.create_user_simulator

    def create_user_simulator(simulator_config, **kwargs):
        if isinstance(simulator_config, LocalSimulatorConfig):
            return LocalUserSimulator(**kwargs)
        return original_factory(simulator_config, **kwargs)

    config.RunConfig = LocalRunConfig
    factory.create_user_simulator = create_user_simulator
