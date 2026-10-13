// Run with: node tests/test_html_chat_context.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { randomUUID } = require('node:crypto');

const html = fs.readFileSync(path.join(__dirname, '..', 'ai_ui.html'), 'utf8');
const script = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
    .map(m => m[1]).join('\n')
    .replace('        checkServerConnection();', '')
    .replace('        initializeApp();', '');

function harness({ storage = new Map() } = {}) {
    const elements = new Map();
    const element = id => {
        if (!elements.has(id)) {
            const classes = new Set();
            elements.set(id, {
            textContent: '', innerHTML: '', style: {}, children: [{}], value: '',
            classList: { add(value) { classes.add(value); }, remove(value) { classes.delete(value); }, toggle(value, force) { const on = force ?? !classes.has(value); if(on) classes.add(value); else classes.delete(value); return on; }, contains(value) { return classes.has(value); } },
            addEventListener() {}, appendChild(child) { this.children.push(child); }, focus() {}, remove() {}, click() { this.clicked = true; },
            checked: false,
            });
            let markup = '';
            Object.defineProperty(elements.get(id), 'innerHTML', {
                get() { return markup; },
                set(value) { markup = value; elements.get(id).children = []; },
            });
        }
        return elements.get(id);
    };
    const context = vm.createContext({
        console, crypto: { randomUUID }, AbortController, FormData, Blob, URL,
        sampleJDs: JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'sample_data.json'), 'utf8')).jds,
        window: { localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) }, location: { protocol: 'http:', port: '8010', origin: 'http://localhost:8010' }, confirm: () => true },
        document: { getElementById: element, querySelectorAll: selector => ({ '.view': ['view-placement','view-upload','view-jobs','view-compare','view-settings'], '.placement-step': ['view-setup','view-matches','view-gap'], '.nav button': ['nav-placement','nav-upload','nav-jobs','nav-compare','nav-settings'] }[selector] || []).map(element), createElement: tagName => Object.assign(element(randomUUID()), { tagName }) },
        setTimeout: () => 0, setInterval() {}, clearTimeout() {},
        fetch: async () => { throw new Error('Unexpected request'); },
    });
    vm.runInContext(script, context);
    return { context, elements, storage, run: code => vm.runInContext(code, context) };
}

const response = data => ({ ok: true, json: async () => data });

module.exports = { harness, response };

if (require.main === module) (async () => {
    const h = harness();
    const calls = [];
    h.context.fetch = async (url, options) => {
        calls.push({ url, body: JSON.parse(options.body) });
        if (url.endsWith('/jds')) return response({ jd_id: 'stored-fullstack' });
        if (url.endsWith('/analyze')) return response({
            resume_id: 'res_fullstack', jd_id: 'stored-fullstack', context_sufficient: true,
            match_score: 70, summary: 'Actual result', skill_gap: { missing_essential: ['Docker'] },
            strengths: [], weaknesses: [], recommendations: [], requirement_match: [],
        });
        return response({ context_id: 'a'.repeat(64), chunk_count: 3 });
    };
    // Use the actual sample IDs rather than a parallel fixture with different fields.
    h.run("resumes = [sampleResumes[4]]; resumes[0].id = 'res_fullstack'; resumes[0].pages = [{text: resumes[0].text, page: 3}]; jds = [sampleJDs[4]];");
    await h.run("loadAnalysis('res_fullstack', jds[0].id)");
    assert.equal(calls[0].body.jd_text, h.run('jds[0].text'));
    assert.equal(calls[1].body.jd_id, 'stored-fullstack');
    await h.run('ensureChatContextIndexed()');
    const payload = calls[2].body;
    assert.equal(payload.jd.source, 'Full Stack Software Engineer');
    assert.equal(payload.jd.pages[0].text, h.run('jds[0].text'));
    assert.equal(payload.resume.pages[0].page, 3);
    assert.deepEqual(payload.analysis.skill_gap.missing_essential, ['Docker']);
    assert.equal(payload.analysis.jd_id, payload.jd.backend_id);
    h.context.completedAnalysis = h.run('currentJdData');
    h.run("chatHistory = [{role: 'user', content: 'Old question'}]; updateChatContext();");
    assert.equal(h.run('chatHistory.length'), 1, 'Reopening unchanged context preserves history');
    h.run("resumes.push({...sampleResumes[0], id: 'another'}); generateRankings = async () => {}; onCandidateSelected('another');");
    assert.equal(h.run('chatContextId'), null);
    assert.equal(h.run('chatHistory.length'), 0);
    assert.equal(h.run('buildChatContextPayload().jd'), null);

    // A late indexing response must never attach the old candidate's context.
    h.run("currentResumeId = 'res_fullstack'; selectedJdId = jds[0].id; currentJdData = completedAnalysis; updateChatContext();");
    let resolveIndex;
    h.context.fetch = () => new Promise(resolve => { resolveIndex = resolve; });
    const indexing = h.run('ensureChatContextIndexed()');
    h.run("onCandidateSelected('res_fullstack')");
    resolveIndex(response({ context_id: 'b'.repeat(64), chunk_count: 1 }));
    assert.equal(await indexing, false);
    assert.equal(h.run('chatContextId'), null);
    assert.equal(h.run('chatIsIndexing'), false);

    // Provider/API errors must not turn into a randomly scored successful analysis.
    h.run('analysisCache.clear()');
    h.context.fetch = async () => ({ ok: false, status: 503, json: async () => ({ detail: 'Provider unavailable' }) });
    await h.run("loadAnalysis('res_fullstack', jds[0].id)");
    assert.equal(h.run('currentJdData'), null);
    assert.match(h.elements.get('status-banner-msg').textContent, /Provider unavailable/);
    assert.equal(h.run('buildChatContextPayload().jd.source'), 'Full Stack Software Engineer');

    // Responses for a previous conversation must not appear in the new one.
    h.run("currentResumeId = 'res_fullstack'; selectedJdId = jds[0].id; currentJdData = completedAnalysis; updateChatContext();");
    let resolveAnswer;
    h.context.fetch = async url => url.endsWith('/chat/documents')
        ? response({ context_id: 'c'.repeat(64), chunk_count: 2 })
        : new Promise(resolve => { resolveAnswer = resolve; });
    const sending = h.run("sendChatMessageText('What experience is listed?')");
    while (!resolveAnswer) await new Promise(resolve => setImmediate(resolve));
    h.run("onCandidateSelected('another')");
    resolveAnswer(response({ answer: 'Old candidate answer', context_sufficient: true, citations: [] }));
    await sending;
    assert.equal(h.run('chatHistory.length'), 0);
    assert.equal(h.run('chatIsStreaming'), false);
    console.log('HTML chat context regressions passed: complete evidence, page references, context switches, stale responses, and real analysis errors.');
})().catch(error => { console.error(error); process.exitCode = 1; });
