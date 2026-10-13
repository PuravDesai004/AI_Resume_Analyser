// Run with: node tests/test_html_chat_gaps.cjs
const assert = require('node:assert/strict');
const { harness, response } = require('./test_html_chat_context.cjs');

function selectedPair() {
    const h = harness();
    h.run("jds = [{...sampleJDs[0], id: 'stored-python', backend_id: 'stored-python'}]; onCandidateSelected('res_devops_eng'); selectedJdId = 'stored-python'; updateChatContext();");
    return h;
}
const analysis = (resume_id, jd_id) => ({
    resume_id, jd_id, context_sufficient: true, match_score: 20,
    summary: 'The resume demonstrates Docker; Python, FastAPI and PostgreSQL are not evidenced.',
    strengths: [], weaknesses: [], recommendations: [], requirement_match: [],
    skill_gap: { matched: [{skill_name: 'Docker', match_type: 'exact'}], missing_essential: ['Python', 'FastAPI', 'PostgreSQL'], missing_preferred: [], extra: [] },
});

(async () => {
    const h = selectedPair();
    const calls = [];
    h.context.fetch = async (url, options) => {
        if (options.method === 'DELETE') return response({ deleted_chunks: 3 });
        const body = JSON.parse(options.body);
        calls.push({ url, body });
        if (url.endsWith('/analyze')) return response(analysis(body.resume_id, body.jd_id));
        if (url.endsWith('/chat/documents')) return response({ context_id: 'a'.repeat(64), chunk_count: 3 });
        if (url.endsWith('/chat')) return response({ answer: 'Python, FastAPI and PostgreSQL are not evidenced in this resume.', context_sufficient: true, citations: [] });
        throw new Error(`Unexpected request ${url}`);
    };
    assert.equal(h.run('openChatView()'), false, 'Chat must not open before analysis');
    assert.equal(h.run('chatPanelOpen'), false);
    assert.equal(await h.run('ensureChatContextIndexed()'), false);
    await h.run("sendChatMessageText('Explain the missing essential skills.')");
    assert.equal(calls.length, 0, 'Unanalysed pairs must not index or send chat');
    assert.equal(h.run('chatHistory.length'), 0);

    await h.run("loadAnalysis(currentResumeId, selectedJdId)");
    assert.equal(h.elements.get('nav-placement').classList.contains('active'), true);
    assert.equal(h.elements.get('view-gap').classList.contains('active'), true);
    assert.equal(h.elements.get('ask-ai-btn').disabled, false);
    assert.equal(h.run('openChatView()'), true);
    assert.equal(h.run('chatPanelOpen'), true);
    await h.run("sendChatMessageText('Explain the missing essential skills.')");
    assert.deepEqual(calls.map(c => c.url.split('/').at(-1)), ['analyze', 'documents', 'chat']);
    assert.equal(calls[0].body.resume_id, 'res_devops_eng');
    assert.equal(calls[0].body.jd_id, 'stored-python');
    assert.deepEqual(calls[1].body.analysis.skill_gap.missing_essential, ['Python', 'FastAPI', 'PostgreSQL']);
    assert.equal(h.run('chatHistory.length'), 2);
    h.run('closeChatPanel(); openChatView()');
    assert.equal(h.run('chatHistory.length'), 2, 'Reopening the same report preserves conversation');
    await h.run("sendChatMessageText('What projects help with these gaps?')");
    assert.equal(calls.filter(c => c.url.endsWith('/analyze')).length, 1);
    assert.equal(calls.filter(c => c.url.endsWith('/chat/documents')).length, 1);
    assert.equal(calls.at(-1).body.history.length, 2);

    const oldSession = h.run('CHAT_SESSION_ID');
    await h.run('clearChat()');
    assert.notEqual(h.run('CHAT_SESSION_ID'), oldSession);
    assert.equal(h.run('chatPanelOpen'), true, 'Clearing stays in the same report chat');
    assert.equal(h.run('chatHistory.length'), 0);
    assert.equal(h.run('currentJdData.context_sufficient'), true);

    // Changing the resume locks chat and discards questions for the previous match.
    h.run("onCandidateSelected('res_frontend_eng')");
    assert.equal(h.run('chatPanelOpen'), false);
    assert.equal(h.run('chatHistory.length'), 0);
    assert.equal(h.elements.get('ask-ai-btn').disabled, true);
    assert.equal(h.elements.get('view-setup').classList.contains('active'), true);
    assert.equal(h.run('openChatView()'), false);

    const failure = selectedPair();
    failure.context.fetch = async () => ({ ok: false, status: 503, json: async () => ({ detail: 'Provider unavailable' }) });
    await failure.run('loadAnalysis(currentResumeId, selectedJdId)');
    assert.equal(failure.run('currentJdData'), null);
    assert.equal(failure.run('openChatView()'), false, 'Failed reports must not unlock chat');
    assert.equal(failure.elements.get('ask-ai-btn').disabled, true);

    const mismatched = selectedPair();
    mismatched.context.otherAnalysis = analysis('res_python_dev', 'stored-python');
    mismatched.run('currentJdData = otherAnalysis; updateChatContext()');
    assert.equal(mismatched.run('openChatView()'), false, 'A report for another resume must not unlock chat');
    mismatched.context.otherAnalysis = analysis('res_devops_eng', 'different-job');
    mismatched.run('currentJdData = otherAnalysis; updateChatContext()');
    assert.equal(mismatched.run('openChatView()'), false, 'A report for another job must not unlock chat');

    const indexingFailure = selectedPair();
    indexingFailure.context.goodAnalysis = analysis('res_devops_eng', 'stored-python');
    indexingFailure.run('currentJdData = goodAnalysis; updateChatContext(); openChatView()');
    indexingFailure.context.fetch = async () => ({ ok:false, status:503, json:async()=>({detail:'Embedding provider unavailable'}) });
    await indexingFailure.run("sendChatMessageText('Explain my gaps')");
    assert.match(indexingFailure.elements.get('chat-status-bar').textContent, /Embedding provider unavailable/);
    assert.equal(indexingFailure.run('chatIsStreaming || chatIsIndexing'), false, 'Failures must allow retry');
    assert.equal(indexingFailure.run('chatContextId'), null);
    indexingFailure.context.fetch = async url => url.endsWith('/chat/documents')
        ? response({ context_id:'b'.repeat(64), chunk_count:3 })
        : response({ answer:'Retry succeeded', context_sufficient:true, citations:[] });
    await indexingFailure.run("sendChatMessageText('Explain my gaps')");
    assert.equal(indexingFailure.run('chatHistory.at(-1).content'), 'Retry succeeded');

    // Refresh keeps uploaded resumes and the choice, while requiring a fresh analysis.
    const storage = new Map();
    const original = harness({ storage });
    original.run("resumes = [{ id:'personal', name:'My resume', text:'Resume text', pages:[{text:'Resume text',page:2}] }]; onCandidateSelected('personal');");
    const refreshed = harness({ storage });
    assert.equal(refreshed.run('currentResumeId'), 'personal');
    assert.equal(refreshed.run('resumes[0].pages[0].page'), 2);
    assert.equal(refreshed.run('openChatView()'), false);
    original.run('resumes = []; currentResumeId = null; saveResumeState()');
    assert.equal(harness({ storage }).run('resumes.length'), 0, 'Reset state survives refresh');
    storage.set('resume-analyzer-resumes-v1', 'bad json');
    assert.equal(harness({ storage }).run('resumes.length'), 5, 'Corrupt storage must not break startup');
    console.log('Report-first flow passed: chat gates, report reuse, resume switches, mismatched/failed analysis and refresh persistence.');
})().catch(error => { console.error(error); process.exitCode = 1; });
