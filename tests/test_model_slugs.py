import pathlib
import re
import unittest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "vox_novel"

# Slugs OpenRouter no longer serves. Each was a default or an offered choice here
# at some point, and each failed at runtime with HTTP 404 rather than at startup.
WITHDRAWN = {
    "minimax/minimax-m3:free",
    "minimax/minimax-m2.7:free",
    "z-ai/glm-5.2:free",
}

# A model id as it appears in code: two or more path segments, optionally :free.
SLUG = re.compile(r"\b[a-z0-9][\w.-]*/[\w.-]+(?::free)?\b")


def source_files():
    for pattern in ("**/*.py", "**/*.html"):
        for path in SRC.glob(pattern):
            if "__pycache__" not in path.parts:
                yield path


class TestNoWithdrawnModelSlugs(unittest.TestCase):
    """A withdrawn slug fails at request time, not at startup.

    Nothing validates a configured model, so a dead default surfaces as a 404 in
    the middle of translating a chapter -- or, for voice prompts, as a silent
    downgrade to the keyword table. Catching it here is the cheap place.
    """

    def test_no_withdrawn_slug_is_referenced(self):
        offenders = []
        for path in source_files():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for dead in WITHDRAWN:
                    if dead in line:
                        offenders.append(f"{path.relative_to(SRC)}:{lineno}: {dead}")
        self.assertEqual(
            offenders,
            [],
            "withdrawn model slugs are still referenced:\n  " + "\n  ".join(offenders),
        )

    def test_the_default_model_is_not_withdrawn(self):
        from vox_novel.settings import get_app_settings

        self.assertNotIn(get_app_settings()["openrouter_model"], WITHDRAWN)

    def test_no_hardcoded_fallback_to_a_withdrawn_slug(self):
        from vox_novel.translators.openrouter import OpenRouterTranslator

        for model in ("minimax/minimax-m3", "google/gemma-4-31b-it:free", "deepseek/deepseek-v4-pro"):
            t = OpenRouterTranslator(model=model, api_key="test")
            self.assertFalse(
                set(t.fallback_models) & WITHDRAWN,
                f"{model} falls back to a withdrawn slug: {t.fallback_models}",
            )


class TestOfferedChoicesAreConsistent(unittest.TestCase):
    """The CLI picker maps a chosen label to a model id by substring match."""

    def test_every_cli_choice_maps_to_a_live_model(self):
        cli = (SRC / "cli.py").read_text(encoding="utf-8")
        offered = set(SLUG.findall(cli)) & {
            s for s in SLUG.findall(cli) if "/" in s and not s.startswith(("http", "vox_novel"))
        }
        self.assertFalse(
            offered & WITHDRAWN, f"the CLI offers withdrawn models: {offered & WITHDRAWN}"
        )

    def test_settings_modal_offers_no_withdrawn_model(self):
        modal = (SRC / "web" / "templates" / "settings_modal.html").read_text(encoding="utf-8")
        found = {v for v in re.findall(r'<option value="([^"]+)"', modal)}
        self.assertFalse(
            found & WITHDRAWN, f"the settings modal offers withdrawn models: {found & WITHDRAWN}"
        )


if __name__ == "__main__":
    unittest.main()
