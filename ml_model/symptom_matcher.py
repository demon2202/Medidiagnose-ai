import re
import numpy as np

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_FUZZ = True
except Exception:
    HAS_FUZZ = False


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


class SymptomMatcher:
    def __init__(self, symptoms, synonyms=None):
        self.symptoms = list(symptoms)
        self.synonyms = dict(synonyms or {})
        self._name_to_canon = {}
        self._alias_to_canon = {}
        self._labels = {}

        for s in self.symptoms:
            name = _norm(s.replace("_", " "))
            self._name_to_canon[name] = s
            self._labels[s] = s.replace("_", " ").title()
        # reverse synonym map: alias -> canonical
        for alias, canon in self.synonyms.items():
            if canon in set(self.symptoms):
                self._alias_to_canon[_norm(alias)] = canon

        # TF-IDF corpus: each symptom doc = name + its aliases
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        docs = []
        self._doc_index = []
        for s in self.symptoms:
            aliases = [_norm(a) for a, c in self.synonyms.items() if c == s]
            doc = self._labels[s] + " " + " ".join(aliases)
            docs.append(doc)
            self._doc_index.append(s)
        self._vectors = self._vec.fit_transform(docs)

    def match_terms(self, query, fuzzy_threshold=78, tfidf_threshold=0.30):
        """Split a free-text query into terms and match each term.

        Returns (matches, unmatched): matches is a list of dicts
        {canonical, label, score, method}; unmatched is a list of strings.
        """
        terms = self._split(query)
        matches, unmatched = [], []
        seen = set()
        for term in terms:
            m = self.match_one(term, fuzzy_threshold, tfidf_threshold)
            if m is None:
                unmatched.append(term)
            else:
                if m["canonical"] not in seen:
                    seen.add(m["canonical"])
                    matches.append(m)
        return matches, unmatched

    _LEAD = re.compile(
        r"^(?:i\s+(?:am|have|got|feel|feeling|suffer|suffering|experienc\w+|noticing|notice)\s+"
        r"|i'?m\s+|my\s+|the\s+|a\s+|an\s+|really\s+|very\s+|quite\s+|severe\s+|bad\s+|"
        r"a\s+bit\s+|some\s+|lots\s+of\s+|constant\s+|having\s+|with\s+)+",
        re.IGNORECASE,
    )

    @classmethod
    def _strip_lead(cls, term):
        while True:
            new = cls._LEAD.sub("", term, count=1)
            if new == term:
                return new.strip()
            term = new

    @classmethod
    def _split(cls, query):
        q = str(query).lower().strip()
        if not q:
            return []
        # split on common separators BEFORE normalizing (commas are removed by
        # _norm, so we must split the raw text first)
        parts = re.split(r",| and |\band\b|&|;|\+| or |\bor\b|\n|\.", q)
        out = []
        for p in parts:
            p = _norm(p)
            p = cls._strip_lead(p)
            if p:
                out.append(p)
        return out

    def match_one(self, term, fuzzy_threshold=78, tfidf_threshold=0.30):
        t = _norm(term)
        if not t:
            return None

        # 1. exact canonical name
        if t in self._name_to_canon:
            c = self._name_to_canon[t]
            return {"canonical": c, "label": self._labels[c], "score": 100.0, "method": "exact"}

        # 2. synonym alias
        if t in self._alias_to_canon:
            c = self._alias_to_canon[t]
            return {"canonical": c, "label": self._labels[c], "score": 99.0, "method": "synonym"}

        # 3. fuzzy against all known names + aliases
        if HAS_FUZZ:
            candidates = list(self._name_to_canon) + list(self._alias_to_canon)
            best = rf_process.extractOne(
                t, candidates, scorer=fuzz.ratio, score_cutoff=fuzzy_threshold
            )
            if best is not None:
                match_str, score, _ = best
                c = self._name_to_canon.get(match_str) or self._alias_to_canon.get(match_str)
                return {"canonical": c, "label": self._labels[c],
                        "score": float(score), "method": "fuzzy"}

        # 4. TF-IDF cosine (sparse embedding) over the whole term as a phrase
        qv = self._vec.transform([t])
        sims = np.asarray(qv.dot(self._vectors.T).todense()).ravel()
        idx = int(np.argmax(sims))
        if sims[idx] >= tfidf_threshold:
            c = self._doc_index[idx]
            return {"canonical": c, "label": self._labels[c],
                    "score": float(sims[idx] * 100), "method": "tfidf"}

        return None


# Extra colloquial aliases layered on top of the server's own synonym map.
# (The server's SYMPTOM_SYNONYMS is much larger; these add common phrasings
# that fuzzy/TF-IDF can't catch, e.g. "throwing up" -> vomiting.)
COMMON_ALIASES = {
    "fever": "high_fever",
    "throwing up": "vomiting",
    "throw up": "vomiting",
    "puking": "vomiting",
    "feeling sick": "nausea",
    "sick to my stomach": "nausea",
    "high temperature": "high_fever",
    "running a temperature": "high_fever",
    "feverish": "high_fever",
    "low grade fever": "mild_fever",
    "slight fever": "mild_fever",
    "coughing": "cough",
    "dry cough": "cough",
    "wet cough": "cough",
    "can't breathe": "breathlessness",
    "short of breath": "breathlessness",
    "out of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "sore throat": "throat_irritation",
    "scratchy throat": "throat_irritation",
    "head ache": "headache",
    "migrane": "headache",
    "stomach ache": "stomach_pain",
    "tummy ache": "stomach_pain",
    "belly ache": "belly_pain",
    "upset stomach": "indigestion",
    "loose motion": "diarrhoea",
    "loose motions": "diarrhoea",
    "the runs": "diarrhoea",
    "constipated": "constipation",
    "dizzy": "dizziness",
    "light headed": "dizziness",
    "lightheaded": "dizziness",
    "feeling faint": "dizziness",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "worn out": "fatigue",
    "sleepy": "lethargy",
    "itchy": "itching",
    "itch": "itching",
    "rash": "skin_rash",
    "hives": "skin_rash",
    "sneezing": "continuous_sneezing",
    "runny nose": "runny_nose",
    "stuffy nose": "congestion",
    "blocked nose": "congestion",
    "chest pain": "chest_pain",
    "heart racing": "fast_heart_rate",
    "palpitation": "palpitations",
    "heart beating fast": "fast_heart_rate",
    "loss of appetite": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "weight loss": "weight_loss",
    "losing weight": "weight_loss",
    "weight gain": "weight_gain",
    "gaining weight": "weight_gain",
    "yellow eyes": "yellowing_of_eyes",
    "yellow skin": "yellowish_skin",
    "jaundiced": "yellowish_skin",
    "dark urine": "dark_urine",
    "blood in stool": "bloody_stool",
    "blood in urine": "blood_in_sputum",
    "coughing up blood": "blood_in_sputum",
    "joint pain": "joint_pain",
    "aching joints": "joint_pain",
    "muscle ache": "muscle_pain",
    "body ache": "muscle_pain",
    "body aches": "muscle_pain",
    "back ache": "back_pain",
    "neck ache": "neck_pain",
    "knee ache": "knee_pain",
    "blurred vision": "blurred_and_distorted_vision",
    "blurry vision": "blurred_and_distorted_vision",
    "watery eyes": "watering_from_eyes",
    "red eyes": "redness_of_eyes",
    "sweating": "sweating",
    "night sweats": "sweating",
    "shaking": "shivering",
    "chills": "chills",
    "swollen glands": "swelled_lymph_nodes",
    "swollen lymph nodes": "swelled_lymph_nodes",
    "swollen ankles": "swollen_legs",
    "swollen feet": "swollen_legs",
    "numbness": "altered_sensorium",
    "slurring": "slurred_speech",
    "slurred speech": "slurred_speech",
    "trouble speaking": "slurred_speech",
    "weakness one side": "weakness_of_one_body_side",
    "weakness in limbs": "weakness_in_limbs",
    "frequent urination": "polyuria",
    "burning urination": "burning_micturition",
    "burning when peeing": "burning_micturition",
    "bad smelling urine": "foul_smell_of_urine",
    "acidity": "acidity",
    "heartburn": "acidity",
    "acid reflux": "acidity",
    "gas": "passage_of_gases",
    "bloating": "distention_of_abdomen",
    "cramping": "cramps",
    "thirsty": "dehydration",
    "very thirsty": "dehydration",
    "drinking lots of water": "excessive_hunger",
    "always hungry": "excessive_hunger",
    "mood swings": "mood_swings",
    "feeling down": "depression",
    "sad": "depression",
    "anxious": "anxiety",
    "worried": "anxiety",
    "can't concentrate": "lack_of_concentration",
    "trouble concentrating": "lack_of_concentration",
    "forgetful": "lack_of_concentration",
    "vomiting blood": "stomach_bleeding",
    "pale skin": "yellowish_skin",
    "bruising easily": "bruising",
    "hair loss": "loss_of_smell",
}


if __name__ == "__main__":
    import json
    symptoms = json.load(open(r"C:\Users\Harsh\OneDrive\Desktop\Harshit\medidiagnose-ai\ml_model\symptom_list.json", encoding="utf-8"))
    m = SymptomMatcher(symptoms, COMMON_ALIASES)
    tests = [
        "fever and a bad cough",
        "caugh",
        "throwing up",
        "I feel really dizzy and tired",
        "itchy rash on my arm",
        "short of breath",
        "chest pain",
        "fever, headache, body ache",
        "loose motions",
        "swollen glands",
        "burning when I pee",
        "heart racing",
    ]
    for t in tests:
        matches, unmatched = m.match_terms(t)
        print(f"\nQUERY: {t!r}")
        for mm in matches:
            print(f"   -> {mm['canonical']:30s} ({mm['label']}) [{mm['method']} {mm['score']:.0f}]")
        if unmatched:
            print(f"   unmatched: {unmatched}")
