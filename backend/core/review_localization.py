"""Translate known application messages, never arbitrary source evidence."""
_ROWS = '''Summary analysis failed before all document evidence could be summarized. Please retry; no complete summary was produced.|ஆவணத்தின் முழு ஆதாரத்தையும் சுருக்க முடியவில்லை. முழுமையான சுருக்கம் உருவாகவில்லை; மீண்டும் முயற்சிக்கவும்.|सभी साक्ष्यों का सारांश नहीं बन सका। पूर्ण सारांश तैयार नहीं हुआ; दोबारा प्रयास करें।
AI answer generation failed. The retrieved source excerpts remain available.|AI பதிலை உருவாக்க முடியவில்லை. மீட்டெடுத்த மூல உரைகளைப் பார்க்கலாம்.|AI उत्तर नहीं बन सका। प्राप्त मूल अंश उपलब्ध हैं।
Party not deterministically identified|விதிகளால் தரப்பை அடையாளம் காண முடியவில்லை|नियमों से पक्ष की पहचान नहीं हो सकी
Not specified in extracted sentence|பிரித்தெடுத்த வாக்கியத்தில் குறிப்பிடப்படவில்லை|निकाले गए वाक्य में निर्दिष्ट नहीं है
AI extraction is unavailable; showing rule-based candidate sentences.|AI பிரித்தெடுத்தல் கிடைக்கவில்லை; விதிகளால் தேர்ந்தெடுத்த வாக்கியங்கள் காட்டப்படுகின்றன.|AI निष्कर्षण उपलब्ध नहीं है; नियमों से चुने संभावित वाक्य दिखाए जा रहे हैं।
No readable document evidence is available.|படிக்கக்கூடிய ஆவண ஆதாரம் இல்லை.|पढ़ने योग्य दस्तावेज़ साक्ष्य उपलब्ध नहीं है।
Classification is not a calibrated probability; confirm the category.|வகைப்பாடு அளவிடப்பட்ட நிகழ்தகவு அல்ல; வகையை உறுதிசெய்யவும்.|वर्गीकरण मापी गई संभाव्यता नहीं है; श्रेणी की पुष्टि करें।
The AI service could not complete the negotiation analysis.|AI சேவையால் பேச்சுவார்த்தை ஆய்வை முடிக்க முடியவில்லை.|AI सेवा बातचीत की समीक्षा पूरी नहीं कर सकी।
Not assessed.|மதிப்பிடப்படவில்லை.|आकलन नहीं हुआ।
Not generated.|உருவாக்கப்படவில்லை.|तैयार नहीं हुआ।
Not generated because analysis failed.|ஆய்வு தோல்வியால் உருவாக்கப்படவில்லை.|समीक्षा विफल होने के कारण तैयार नहीं हुआ।
No automated risk rating was produced because an LLM is not configured.|LLM அமைக்கப்படாததால் தானியங்கி அபாய மதிப்பீடு உருவாகவில்லை.|LLM कॉन्फ़िगर नहीं है, इसलिए स्वचालित जोखिम रेटिंग नहीं बनी।
Review the cited excerpt with qualified legal counsel.|குறிப்பிட்ட ஆதாரத்தைத் தகுதியான சட்ட நிபுணருடன் ஆய்வு செய்யவும்.|उद्धृत अंश की योग्य कानूनी विशेषज्ञ के साथ समीक्षा करें।
The LLM could not analyze this category.|LLM இந்த வகையை ஆய்வு செய்ய முடியவில்லை.|LLM इस श्रेणी की समीक्षा नहीं कर सका।
Check provider availability or use the cited excerpt for manual review.|AI சேவை கிடைப்பைச் சரிபார்க்கவும் அல்லது குறிப்பிட்ட ஆதாரத்தை நேரடியாக ஆய்வு செய்யவும்.|AI सेवा की उपलब्धता जाँचें या उद्धृत अंश की मानवीय समीक्षा करें।'''
MESSAGES={}
for row in _ROWS.splitlines():
    en,ta,hi=row.split('|');MESSAGES[en]={'ta':ta,'hi':hi}
# A source quotation may equal a UI phrase. Translate only designated presentation fields.
FIELDS={'summary','answer','party','deadline_frequency','consequence','assessment','potential_impact',
        'negotiation_tactic','safer_alternative','confidence_note','why_risky','recommendation','message','error'}

def localize_payload(value,language,key=None):
    if isinstance(value,dict):
        return {k:localize_payload(v,language,k) for k,v in value.items()}
    if isinstance(value,list):return [localize_payload(v,language,key) for v in value]
    if key in FIELDS and isinstance(value,str):return MESSAGES.get(value,{}).get(language,value)
    return value
