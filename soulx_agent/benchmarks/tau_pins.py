"""Restore canonical Voice V1 reloads while retaining exact upstream hash checks."""


def install():
    import tau2.metrics.benchmark as metrics
    import tau2.multilingual.runtime as runtime
    import tau2.runner.simulation as simulation
    from tau2.runner.presets import binding_revision, load_cell_language

    if getattr(runtime.reload_call_language, "soulx_canonical_pins", False):
        return
    original_reload = runtime.reload_call_language
    original_authoritative = metrics._authoritative_language

    def reload(context, domain):
        try:
            return original_reload(context, domain)
        except ValueError:
            if context.language != "en" or context.mode != "voice":
                raise
            loaded = runtime.load_language(
                context.language,
                domain,
                mode=context.mode,
                localize_db=context.scenario_mode != "canonical",
                selection=context.task_selection,
                pack_extension=context.pack_extension,
                canonical_tasks=True,
            )
            if loaded.artifact_pins != context.artifact_pins:
                raise ValueError("Canonical Voice V1 artifacts differ from the recorded call")
            return loaded

    def authoritative(language, domain, expected_binding, mode):
        try:
            return original_authoritative(language, domain, expected_binding, mode)
        except ValueError:
            if language != "en" or mode != "voice":
                raise
            loaded = load_cell_language(language, domain, mode, canonical_tasks=True)
            if binding_revision(loaded) != expected_binding:
                raise ValueError("Canonical Voice V1 binding differs from the recorded benchmark")
            return loaded

    reload.soulx_canonical_pins = True
    runtime.reload_call_language = reload
    simulation.reload_call_language = reload
    metrics._authoritative_language = authoritative
