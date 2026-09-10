import json
import os
import time
from datetime import datetime
from typing import Dict, List

from openai import OpenAI

from models.abstract_model import AbstractModel
from models.prompts import PROMPTS, PROMPT_VARIANTS, VERIFY_TEMPLATE, VERIFY_META


class OpenRouterModel(AbstractModel):
    def __init__(self, model_name: str = "qwen/qwen3.6-27b", temperature: float = 0.0,
                 prompt_style: str = "simple", verify: bool = False, verbose: bool = False,
                 log_name: str = ""):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_style = prompt_style
        self.verify = verify
        self.verbose = verbose
        # One log file per model instance covering the full run
        os.makedirs("logs", exist_ok=True)
        fname = f"{log_name}.jsonl" if log_name else f"openrouter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._log_path = os.path.join("logs", fname)
        self._log_file = open(self._log_path, "a", encoding="utf-8")
        print(f"[raw log → {self._log_path}]", flush=True)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "https://github.com/lm-kbc-2026",
                "X-Title": "LM-KBC 2026",
            },
        )

    def generate_predictions(self, inputs: List[Dict[str, str]]) -> List[List[str]]:
        results = []
        for row in inputs:
            subject = row["SubjectEntity"]
            relation = row["Relation"]
            prompt = self.build_prompt(subject, relation)
            if relation == "awardWonBy":
                max_tokens = 16000
            elif relation in ("hasArea", "hasCapacity"):
                max_tokens = 8000
            else:
                max_tokens = 8000
            response = self.call_openrouter(prompt, max_tokens=max_tokens)
            self._last_raw = response
            think_part = response.split("</think>")[0] if "</think>" in response else ""
            answer_part = response.split("</think>")[-1] if "</think>" in response else response
            self._log_file.write(json.dumps({
                "subject": subject, "relation": relation,
                "think_chars": len(think_part), "answer_chars": len(answer_part),
                "raw": response
            }, ensure_ascii=False) + "\n")
            self._log_file.flush()
            if self.verbose:
                print(f"\n--- RAW ({len(response)} chars, think={len(think_part)} answer={len(answer_part)}) ---", flush=True)
                print(response, flush=True)
                print("--- END RAW ---\n", flush=True)
            predictions = self.parse_response(response, relation)
            # Thinking-loop recovery: ONLY for awardWonBy, where the model reasons through a
            # long recipient list and the regex mines person names. For other relations an
            # empty answer is a valid signal (e.g. companyTrades [] = not publicly listed) and
            # scraping the thinking text yields garbage (prompt fragments, meta-commentary).
            if not predictions and think_part and relation == "awardWonBy":
                predictions = self._extract_from_thinking(think_part, subject, relation)
            if self.verify and relation in VERIFY_META and predictions:
                verified = self._verify(subject, relation, predictions[0])
                if verified:
                    predictions = verified
            results.append(predictions)
            time.sleep(0.5)
        return results

    def build_prompt(self, subject: str, relation: str) -> str:
        if self.prompt_style != "simple":
            tmpl = PROMPT_VARIANTS.get(relation, {}).get(self.prompt_style)
            if tmpl:
                return tmpl.format(subject=subject)
        return PROMPTS[relation].format(subject=subject)

    def _verify(self, subject: str, relation: str, answer: str):
        desc, numtype = VERIFY_META[relation]
        vp = VERIFY_TEMPLATE.format(relation_desc=desc, subject=subject, answer=answer, numtype=numtype)
        raw = self.call_openrouter(vp, max_tokens=600)
        return self.parse_response(raw, relation)

    def call_openrouter(self, prompt: str, max_tokens: int = 3000) -> str:
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )
                if not response.choices:
                    raise ValueError("No choices in response")
                content = response.choices[0].message.content or ""
                # OpenRouter may return reasoning in a separate field — append if present
                reasoning = getattr(response.choices[0].message, "reasoning", None)
                if reasoning and "</think>" not in content:
                    content = f"<think>{reasoning}</think>{content}"
                return content
            except Exception as e:
                err = str(e)
                if ("429" in err or "rate" in err.lower() or "503" in err
                        or "timeout" in err.lower() or "timed out" in err.lower()
                        or "over capacity" in err.lower()
                        or "JSONDecodeError" in type(e).__name__
                        or "Expecting value" in err
                        or "No choices" in err):
                    wait = 10 * (2 ** attempt)  # 10s, 20s, 40s, 80s
                    print(f"[transient error: {err[:60]}... waiting {wait}s]", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("OpenRouter call failed after 4 retries")
        return ""

    def _extract_from_thinking(self, thinking: str, subject: str, relation: str) -> List[str]:
        """Recover an answer from a thinking loop that never emitted JSON.
        Strategy: API-FIRST — a second extraction call understands the text and returns clean
        names. Regex can't tell a name from a prompt fragment, so it's only the FALLBACK for
        when the extraction call itself loops and returns nothing. (Only used for awardWonBy.)"""
        print(f"[thinking loop detected ({len(thinking)} chars) — API extractor first]", flush=True)
        extraction_prompt = (
            f"The following is partial reasoning about which entities satisfy the relation "
            f"'{relation}' for '{subject}'. Extract every entity name that was identified "
            f"as a correct answer and return them as a JSON array of strings.\n\n"
            f"Reasoning:\n{thinking[-6000:]}\n\n"
            f"Return ONLY a JSON array, e.g. [\"Name1\", \"Name2\"]. If none found, return []."
        )
        extracted = self.call_openrouter(extraction_prompt, max_tokens=3000)
        preds = self.parse_response(extracted, relation)
        if preds:
            print(f"[API extractor recovered {len(preds)} items]", flush=True)
            return preds
        # Fallback: the extraction call itself looped (empty) — mine names from the tail via regex
        print(f"[API extractor empty (looped) — regex fallback]", flush=True)
        names = self._regex_names_from_thinking(thinking)
        if names:
            print(f"[regex recovered {len(names)} names from thinking tail]", flush=True)
        return names

    @staticmethod
    def _regex_names_from_thinking(thinking: str) -> List[str]:
        """Mine person/organization names from a thinking block without an API call.
        Reads the TAIL first (models consolidate their final list near the end).
        Matches three formats the model uses: "Name", - Name (year), and 1. Name."""
        import re
        tail = thinking[-8000:]  # where the consolidated list usually lives
        quoted = re.findall(r'"([A-Z][^"\n]{2,60})"', tail)
        bullets = re.findall(r'-\s+([A-Z][A-Za-z.\-\'\s]{3,50}?)(?:\s*\(\d{4}\)|\n|:)', tail)
        numbered = re.findall(r'\d+\.\s+([A-Z][A-Za-z.\-\'\s]{3,50}?)(?:\n|\()', tail)
        noise = {"The", "This", "Note", "Final", "Answer", "Step", "Year", "Award", "Winner",
                 "Recipients", "List", "So", "Now", "Okay", "Wait", "Actually", "Let", "But"}
        seen, out = set(), []
        for name in quoted + bullets + numbered:
            n = name.strip().rstrip(".,;:")
            if n and n.split()[0] not in noise and n.lower() not in seen and len(n) > 3:
                seen.add(n.lower())
                out.append(n)
        return out

    def parse_response(self, response: str, relation: str) -> List[str]:
        if "</think>" in response:
            response = response.split("</think>")[-1]
        elif "<think>" in response:
            print("[WARNING] thinking block truncated — no </think> found, no answer extracted]", flush=True)
            return []
        response = response.strip()

        # Try full valid JSON first
        try:
            start = response.rindex("[")
            end = response.rindex("]") + 1
            parsed = json.loads(response[start:end])
            if isinstance(parsed, list):
                items = [str(item) for item in parsed if item and str(item).strip()]
                return self._filter_numeric(items, relation)
        except (ValueError, json.JSONDecodeError):
            pass

        # Fallback: truncated mid-array — extract partial quoted strings
        if "[" in response:
            start = response.index("[")
            partial = response[start:]
            import re
            names = re.findall(r'"([^"]+)"', partial)
            if names:
                print(f"[WARNING] truncated JSON — recovered {len(names)} partial items]", flush=True)
                return self._filter_numeric(names, relation)

        return []

    def _filter_numeric(self, items: List[str], relation: str) -> List[str]:
        if relation in ("hasArea", "hasCapacity"):
            try:
                from evaluate import try_parse_number
                items = [v for v in items if try_parse_number(v) is not None]
            except ImportError:
                pass
        return items


if __name__ == "__main__":
    model = OpenRouterModel()
    tests = [
        {"SubjectEntity": "France", "Relation": "countryLandBordersCountry"},
        {"SubjectEntity": "Dave Keon", "Relation": "personHasCityOfDeath"},
        {"SubjectEntity": "Frédéric Chopin", "Relation": "personHasCityOfDeath"},
        {"SubjectEntity": "Turing Award", "Relation": "awardWonBy"},
    ]
    for t in tests:
        preds = model.generate_predictions([t])[0]
        print(f"{t['SubjectEntity']}: {preds[:5]}")
