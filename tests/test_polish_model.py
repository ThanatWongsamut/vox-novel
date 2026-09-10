import os
import unittest
from unittest import mock

from vox_novel.pipeline.manager import NovelPipeline
from vox_novel.translators.openrouter import OpenRouterTranslator


class TestPolishTranslatorSelection(unittest.TestCase):
    """The editor pass may run on a different model than the draft.

    Measured on a real chapter slice (Ch.100, paras 46-60, real glossary), with
    costs billed by OpenRouter rather than computed:

        flash draft + flash polish   $0.014/chapter
        flash draft + PRO polish     $0.097/chapter
        PRO draft   + flash polish   $0.117/chapter
        PRO draft   + PRO polish     $0.206/chapter

    Putting the stronger model on polish rather than draft is both cheaper (polish
    batches 25 paragraphs to the draft's 15, so fewer calls) and more effective
    (polish receives the English original alongside the draft, so it can repair
    meaning, and its prompt targets naturalness).
    """

    def draft(self, model="google/gemma-4-31b-it:free"):
        return OpenRouterTranslator(model=model, api_key="test")

    def select(self, draft, env):
        with mock.patch.dict(os.environ, env, clear=False):
            return NovelPipeline._polish_translator(draft, "openrouter", {})

    def test_unset_reuses_the_draft_translator(self):
        d = self.draft()
        self.assertIs(self.select(d, {"OPENROUTER_POLISH_MODEL": ""}), d)

    def test_absent_env_reuses_the_draft_translator(self):
        d = self.draft()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_POLISH_MODEL", None)
            self.assertIs(NovelPipeline._polish_translator(d, "openrouter", {}), d)

    def test_distinct_model_produces_a_separate_translator(self):
        d = self.draft("deepseek/deepseek-v4-flash")
        got = self.select(d, {"OPENROUTER_POLISH_MODEL": "deepseek/deepseek-v4-pro"})
        self.assertIsNot(got, d)
        self.assertEqual(got.model_name, "deepseek/deepseek-v4-pro")
        self.assertEqual(d.model_name, "deepseek/deepseek-v4-flash", "draft must be untouched")

    def test_same_model_reuses_rather_than_rebuilding(self):
        d = self.draft("deepseek/deepseek-v4-pro")
        self.assertIs(self.select(d, {"OPENROUTER_POLISH_MODEL": "deepseek/deepseek-v4-pro"}), d)

    def test_whitespace_only_value_is_ignored(self):
        d = self.draft()
        self.assertIs(self.select(d, {"OPENROUTER_POLISH_MODEL": "   "}), d)

    def test_unconstructable_model_falls_back_to_the_draft(self):
        d = self.draft()
        with mock.patch(
            "vox_novel.pipeline.manager.translator_registry.get_translator",
            side_effect=RuntimeError("boom"),
        ):
            self.assertIs(
                self.select(d, {"OPENROUTER_POLISH_MODEL": "some/model"}),
                d,
                "a bad polish model must not break translation",
            )


class TestControlPromptMerge(unittest.TestCase):
    """Synthesis runs for minutes; a voice edit made meanwhile must survive it."""

    def setUp(self):
        import shutil, tempfile
        from pathlib import Path
        from vox_novel.storage.knowledge import KnowledgeManager

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.km = KnowledgeManager(base_dir=self.tmp)

    def merge(self, job_copy):
        """Invoke the real implementation, not a copy of it."""
        pipeline = NovelPipeline(knowledge_manager=self.km)
        pipeline._persist_derived_control_prompts("s", job_copy)
        return self.km.load_or_init("s")

    def test_a_mid_job_edit_is_not_undone(self):
        k = self.km.load_or_init("s")
        k.update_narrator_voice(
            voice_description="เสียงชายแก่", voice_control_prompt="old male narrator"
        )
        self.km.save(k)
        job_copy = self.km.load_or_init("s")

        edit = self.km.load_or_init("s")
        edit.update_narrator_voice(
            voice_description="เสียงเด็กหญิง", voice_control_prompt="young girl narrator"
        )
        self.km.save(edit)

        out = self.merge(job_copy)
        self.assertEqual(out.narrator_voice_description, "เสียงเด็กหญิง")
        self.assertEqual(
            out.narrator_voice_control_prompt,
            "young girl narrator",
            "the job restored a prompt derived from the superseded description",
        )

    def test_a_derived_prompt_is_persisted_when_nothing_changed(self):
        k = self.km.load_or_init("s")
        k.update_narrator_voice(voice_description="เสียงชายแก่")
        self.km.save(k)

        job_copy = self.km.load_or_init("s")
        job_copy.narrator_voice_control_prompt = "old male narrator"

        out = self.merge(job_copy)
        self.assertEqual(out.narrator_voice_control_prompt, "old male narrator")

    def test_character_prompt_follows_the_same_rule(self):
        k = self.km.load_or_init("s")
        k.add_character("Aria", "อาเรีย")
        k.update_character_voice("Aria", voice_description="เสียงหวานใส")
        self.km.save(k)

        job_copy = self.km.load_or_init("s")
        job_copy.find_character("Aria").voice_control_prompt = "sweet female voice"

        edit = self.km.load_or_init("s")
        edit.update_character_voice("Aria", voice_description="เสียงห้าว ผู้ชาย")
        self.km.save(edit)

        out = self.merge(job_copy)
        self.assertIsNone(
            out.find_character("Aria").voice_control_prompt,
            "a superseded character prompt was restored",
        )


if __name__ == "__main__":
    unittest.main()
