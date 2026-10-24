"""Replace Tau caller synthesis while retaining its simulator, noise and telephony."""


def install():
    import os

    import tau2.agent.base.voice as voice
    import tau2.voice.synthesis.audio_effects.speech_generator as speech_generator
    import tau2.voice.synthesis.synthesize as synthesis
    from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat

    from soulx_agent.local_caller import synthesize_caller_sync

    def local_synthesis(text, provider, provider_config):
        return AudioData(
            data=synthesize_caller_sync(text),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=24000, channels=1),
        )

    synthesis.synthesize_voice = local_synthesis
    voice.synthesize_voice = local_synthesis
    speech_generator.synthesize_voice = local_synthesis

    if os.environ.get("SOULX_BENCHMARK_PHASE") == "inference":
        import tau2.runner.simulation as simulation
        from tau2.evaluator.evaluator import EvaluationType

        original = simulation.evaluate_simulation

        def environment_only(*args, **kwargs):
            kwargs["evaluation_type"] = EvaluationType.ENV
            return original(*args, **kwargs)

        simulation.evaluate_simulation = environment_only
