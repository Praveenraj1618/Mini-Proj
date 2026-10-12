"""Document routing and full-evidence obligation extraction shared by the reviewer."""
import json
import re
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from core.review_profiles import PROFILES, LANGUAGES, infer_profile, detect_languages
from core.document_analysis import contract_pages, evidence_windows


class ReviewWorkflow:
    def set_preferences(self, language='en', category='auto'):
        if language not in LANGUAGES or (category != 'auto' and category not in PROFILES):
            raise ValueError('Unsupported language or category')
        if (language, category) != (self.response_language, self.category_override):
            self.response_language, self.category_override = language, category
            self._classification = None
            self._cached_summary = self._cached_heatmap = self._cached_risk_analysis = self._cached_obligations = None

    @property
    def profile_id(self):
        if self.category_override != 'auto': return self.category_override
        if self._classification: return self._classification['category_id']
        return infer_profile(self.document.full_text)

    @property
    def profile(self):
        return PROFILES[self.profile_id]

    def _classify_document_heuristically(self):
        p = self.profile
        label = p['labels'][self.response_language]
        if self.profile_id == 'services' and self.response_language == 'en' and 'master services agreement' in self.document.full_text[:500].lower():
            label = 'Master Services Agreement (MSA)'
        return dict(category_id=self.profile_id, doc_type=label, confidence=None,
                    confidence_note='Classification is not a calibrated probability; confirm the category.',
                    source='User selected' if self.category_override != 'auto' else 'Rule-based Heuristic',
                    target_risk_focus=p['topics'], negotiation_supported=p['negotiable'],
                    languages=detect_languages(self.document.full_text))

    def classify_document(self):
        from core.legal_reviewer import check_api_key_configured, parse_llm_json_object
        if self._classification: return dict(self._classification)
        result = self._classify_document_heuristically()
        if self.category_override != 'auto':
            self._classification = result
            return dict(result)
        if not check_api_key_configured(): return result
        pages = contract_pages(self.document)
        budget = max(50, 6000 // max(len(pages), 1))
        sample = '\n'.join(f'[Page {p.page_num}] {p.text[:budget]}' for p in pages)[:8000]
        try:
            content = self._invoke_llm([HumanMessage(content='Classify this bounded document sample. Return JSON with category_id and doc_type. '
                'Choose category_id from ' + ', '.join(PROFILES) + '. Use unknown for unclear or mixed documents.\n' + sample)], max_tokens=250)
            item = parse_llm_json_object(content)
            key = item.get('category_id')
            if key not in PROFILES:
                if not isinstance(item.get('doc_type'), str): raise ValueError('Invalid category')
                key = infer_profile(item['doc_type'])
            p = PROFILES[key]
            result.update(category_id=key, doc_type=p['labels'][self.response_language], target_risk_focus=p['topics'],
                          source='LLM', negotiation_supported=p['negotiable'])
            self._classification = result
        except Exception:
            pass
        return dict(result)

    def extract_obligations(self):
        from core.legal_reviewer import check_api_key_configured, parse_llm_json_array, LEGAL_SYSTEM_PROMPT, SUMMARY_CONTEXT_CHARS
        if self._cached_obligations is not None:
            self.obligation_status = dict(mode='ai', diagnostic=None, scope='All readable pages')
            return self._cached_obligations
        docs = [Document(page_content=p.text, metadata={'page_number':p.page_num}) for p in contract_pages(self.document)]
        self.obligation_status = dict(mode='rule_based', diagnostic=None, scope='Candidate sentences only; not exhaustive')
        if not docs:
            self.obligation_status['diagnostic'] = {'message':'No readable document evidence is available.'}
            return []
        if not check_api_key_configured():
            self.obligation_status['diagnostic'] = {'message':'AI extraction is unavailable; showing rule-based candidate sentences.'}
            return self._extract_obligations_deterministically(docs)
        items, seen = [], set()
        try:
            for window in evidence_windows(self.document, min(SUMMARY_CONTEXT_CHARS, 6000)):
                raw = self._invoke_llm([SystemMessage(content=LEGAL_SYSTEM_PROMPT), HumanMessage(content=(
                    'Extract explicit duties or operative/statutory directions as appropriate. Do not turn allegations, '
                    'requests, permissions or history into obligations. Return a JSON array (empty if none) with party, '
                    'obligation, deadline_frequency, consequence, page (integer), evidence (verbatim quote). '
                    'Use unknown where not specified; no inferred deadlines.\n\n' + window))], max_tokens=2200)
                for item in parse_llm_json_array(raw):
                    if not isinstance(item, dict) or any(not isinstance(item.get(k),str) or not item[k].strip()
                        for k in ('party','obligation','deadline_frequency','consequence','evidence')):
                        raise ValueError('Invalid obligation fields')
                    page, quote = item.get('page'), ' '.join(item['evidence'].split())
                    if type(page) is not int or page not in {int(n) for n in re.findall(r'\[Page (\d+)\]', window)}:
                        raise ValueError('Unavailable page citation')
                    source = next(p.text for p in self.document.pages if p.page_num == page)
                    if quote not in ' '.join(source.split()) or quote not in ' '.join(window.split()):
                        raise ValueError('Evidence is not a verbatim source quotation')
                    if (page,quote) not in seen:
                        seen.add((page,quote)); items.append(item)
            self.obligation_status = dict(mode='ai', diagnostic=None, scope='All readable pages')
            self._cached_obligations = items
            return items
        except Exception as error:
            self.obligation_status['diagnostic'] = self._failure_details(error)
            return self._extract_obligations_deterministically(docs)
