/**
 * AI Service — uses Groq API (FREE, fast, no IP restrictions)
 * Groq provides llama3, mixtral etc. for free with generous rate limits.
 * Get your free API key at: https://console.groq.com
 */
import Groq from 'groq-sdk';

// Lazy client — initialized on first use so dotenv is always loaded first
let _groq = null;
function getGroq() {
  if (!_groq) {
    if (!process.env.GROQ_API_KEY) throw new Error('GROQ_API_KEY is not set in .env');
    _groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
  }
  return _groq;
}

// Current active Groq production models (updated May 2025)
const MODELS = [
  'llama-3.3-70b-versatile',   // Best quality — Meta LLaMA 3.3 70B
  'llama-3.1-8b-instant',      // Fastest — Meta LLaMA 3.1 8B
  'qwen/qwen3-32b',             // Backup — Qwen 32B (preview)
];

/**
 * Extracts a JSON object from raw text (AI often wraps JSON in markdown fences)
 */
function extractJSON(text) {
  try {
    const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (fenceMatch) return JSON.parse(fenceMatch[1].trim());

    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start !== -1 && end !== -1) {
      return JSON.parse(text.slice(start, end + 1));
    }
  } catch (e) {
    console.error('JSON parse error:', e.message);
  }
  return null;
}

/**
 * Generates structured medical report data using Groq (FREE).
 * @param {string} transcript - Doctor's raw notes / voice transcript
 * @param {string[]} detectedTags - Tag names from the DOCX template
 * @param {string} modality - e.g. 'USG', 'X-RAY', 'CT/MRI'
 * @returns {Object} JSON object whose keys match detectedTags
 */
export async function generateReportData(transcript, detectedTags, modality) {
  const schemaExample = detectedTags.reduce((acc, tag) => {
    acc[tag] = `<value for ${tag}>`;
    return acc;
  }, {});

  const systemPrompt = `You are an Elite Medical Typist with 25+ years of experience in Radiology and Pathology.
Transform the raw doctor's notes into a professional medical report.

MODALITY: ${modality}

CRITICAL RULES:
- Use proper clinical terminology (e.g. "coarsened echotexture", "costophrenic angles clear").
- If an organ is not mentioned in the notes, provide a standard normal description.
- For organ-specific tags (e.g., 'liver_finding', 'gallbladder_finding', 'pancreas_finding', 'spleen_finding', 'kidneys_finding', 'urinary_bladder_finding', 'prostate_finding'):
  - Write ONLY the finding description (e.g. "is normal in size and echogenicity. No focal lesion seen.")
  - DO NOT include the organ name in the value because it is already printed in the template header.
- For 'impression': always return an array of strings representing the key findings (e.g., ["Moderate Hepatomegaly...", "No free fluid."]).
- For date fields like "reg_date" or "report_date": use today's date if not provided.
- OUTPUT ONLY RAW JSON — no markdown, no explanation, no preamble whatsoever.

REQUIRED JSON SCHEMA (output exactly these keys, no extras):
${JSON.stringify(schemaExample, null, 2)}`;

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: `Raw Transcript:\n${transcript}` }
  ];

  let lastError;
  let isRateLimit = false;
  for (const model of MODELS) {
    try {
      console.log(`🔵 Calling Groq model: ${model}`);

      const response = await getGroq().chat.completions.create({
        model,
        messages,
        max_tokens: 2000,
        temperature: 0.1,
        stream: false,
      });

      const raw = response.choices[0]?.message?.content;
      if (!raw) throw new Error('Empty response from model');

      console.log(`✅ Response received (${raw.length} chars)`);

      const parsed = extractJSON(raw);
      if (!parsed) throw new Error(`Could not parse JSON. Raw output: ${raw.substring(0, 200)}`);

      return parsed;
    } catch (err) {
      console.warn(`⚠️  ${model} failed: ${err.message}`);
      lastError = err;
      // If it's a rate limit or limit exceeded error, flag it
      if (
        err.status === 429 ||
        err.message?.includes('429') ||
        err.message?.toLowerCase().includes('rate limit') ||
        err.message?.toLowerCase().includes('limit exceeded') ||
        err.message?.toLowerCase().includes('quota')
      ) {
        isRateLimit = true;
      }
      
      if (err.message?.includes('401') || err.message?.includes('auth')) {
        throw new Error('Groq API key invalid. Get a free key at https://console.groq.com and set GROQ_API_KEY in .env');
      }
    }
  }

  if (isRateLimit) {
    throw new Error('GROQ_RATE_LIMIT_EXCEEDED');
  }

  throw new Error(`All Groq models failed. Last error: ${lastError?.message}`);
}
