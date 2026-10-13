// Offline API integration checks: node tests/test_html_features.cjs
const assert = require('node:assert/strict');
const { harness, response } = require('./test_html_chat_context.cjs');

(async () => {
    const h = harness();
    const calls = [];
    const config = { gemini_configured: true, gemini_masked: 'masked', groq_configured: false, groq_masked: 'Not Set', generation_model: 'configured-model', embedding_model: 'embedding-model', rerank: false, jd_count: 2, jd_cap: null };
    const jobs = [
        { jd_id: 'stored-a', title: 'Backend role', company: 'Alpha', location: 'London (Remote)', full_text: 'Python and Docker required.' },
        { jd_id: 'stored-b', title: 'Frontend role', company: 'Beta', location: 'Mumbai (On-site)', full_text: 'React and TypeScript required.' }
    ];
    let fail = null;
    h.context.fetch = async (url, options = {}) => {
        const route = url.replace('http://localhost:8010', '');
        const call = { route, method: options.method || 'GET', body: options.body ? JSON.parse(options.body) : undefined };
        calls.push(call);
        if (fail && route.startsWith(fail)) return { ok: false, status: 503, json: async () => ({ detail: 'Provider unavailable' }) };
        if (route === '/api/config') return response(call.method === 'POST' ? { status: 'ok' } : config);
        if (route === '/jds' && call.method === 'GET') return response({ jds: jobs, count: jobs.length, cap: null });
        if (route === '/jds/count') return response({ count: jobs.length, cap: null });
        if (route === '/jds/ids') return response({ ids: jobs.map(j => j.jd_id) });
        if (route === '/rank') return response({ results: [
            { ...jobs[0], snippet: 'Python role', rank: 1, similarity_score: 0 },
            { ...jobs[1], snippet: 'React role', rank: 2, similarity_score: 85 }
        ] });
        if (route === '/analyze') return response({ resume_id: call.body.resume_id, jd_id: call.body.jd_id, context_sufficient: true, match_score: 80, summary: 'Supported analysis', strengths: [], weaknesses: [], recommendations: [], requirement_match: [], skill_gap: { matched: [{ skill_name: 'Python' }], missing_essential: ['Docker'], missing_preferred: [], extra: [] } });
        if (route === '/api/report/markdown') return response({ markdown: '# Placement report\nSupported analysis' });
        if (route.startsWith('/chat/') && call.method === 'DELETE') return response({ deleted_chunks: 3 });
        if (route.startsWith('/jds/') && call.method === 'DELETE') { const index = jobs.findIndex(j => '/jds/' + j.jd_id === route); if (index >= 0) jobs.splice(index, 1); return response({ deleted: true }); }
        if (route === '/jds' && call.method === 'POST') return response({ jd_id: 'new-stored-job' });
        if (route === '/api/load-fixtures/sample') {
            const fixtureJobs = h.context.sampleJDs.map((j, i) => ({ ...j, id: `sample-${i}`, backend_id: `sample-${i}` }));
            for (const job of fixtureJobs) if (!jobs.some(j => j.jd_id === job.backend_id)) jobs.push({ ...job, jd_id: job.backend_id, full_text: job.text });
            return response({ resumes: [{ id: 'fixture-resume', name: 'Fixture candidate', text: 'Fixture resume' }], jds: fixtureJobs, count: 5, stored_count: jobs.length });
        }
        if (route === '/jds' && call.method === 'DELETE') return response({ count: 0 });
        throw new Error(`Unexpected API call: ${call.method} ${route}`);
    };
    await h.run('syncDataset()');
    assert.equal(h.run('serverJDCount'), 2);
    assert.equal(h.context.document.getElementById('jd-count').innerText, '2 jobs');
    assert.equal(h.run('jds[0].backend_id'), 'stored-a');

    await h.run('loadSettings()');
    assert.equal(h.context.document.getElementById('settings-model').value, 'configured-model');
    await h.run('saveSettings()');
    let saved = calls.find(c => c.route === '/api/config' && c.method === 'POST').body;
    assert.equal('gemini_api_key' in saved, false, 'Blank keys must preserve existing credentials');
    assert.equal('groq_api_key' in saved, false);
    assert.equal(saved.generation_model, 'configured-model');
    h.context.document.getElementById('remove-groq-key').checked = true;
    await h.run('saveSettings()');
    saved = calls.filter(c => c.route === '/api/config' && c.method === 'POST').at(-1).body;
    assert.equal(saved.groq_api_key, '', 'Explicit key removal reaches the backend');

    h.context.document.getElementById('rank-top-n').value = '0';
    h.context.document.getElementById('preferred-location').value = 'London';
    h.context.document.getElementById('preferred-arrangement').value = 'Remote';
    await h.run("generateRankings('res_python_dev', false)");
    const rank = calls.find(c => c.route === '/rank').body;
    assert.equal(rank.top_n, 0);
    assert.equal(rank.preferred_location, 'London');
    assert.equal(rank.preferred_arrangement, 'Remote');
    assert.ok([...h.elements.values()].some(el => el.innerHTML.includes('<strong>0%</strong>')), 'A real zero score must stay zero');
    h.context.document.getElementById('filter-location').value = 'London';
    h.run('renderRankings()');
    assert.equal(h.context.document.getElementById('rank-result-count').textContent, '1 of 2 returned roles shown');
    fail = '/rank';
    await assert.rejects(h.run("generateRankings('res_python_dev', false)"), /Provider unavailable/);
    assert.equal(h.run('lastRankings.length'), 0, 'Failure must not produce simulated rankings');
    assert.match(h.context.document.getElementById('ranking-status').textContent, /Provider unavailable/);
    fail = null;

    await h.run("loadAnalysis('res_python_dev', 'stored-a')");
    await h.run('downloadReport()');
    assert.equal(calls.find(c => c.route === '/api/report/markdown').body.jd_title, 'Backend role');
    assert.ok([...h.elements.values()].some(el => el.tagName === 'a' && el.clicked && el.download.endsWith('.md')));
    h.context.document.querySelectorAll = selector => selector === 'input[name="compare-jd"]:checked' ? [{ value: 'stored-a' }, { value: 'stored-b' }] : [];
    const before = calls.filter(c => c.route === '/analyze').length;
    await h.run('compareRoles()');
    assert.equal(calls.filter(c => c.route === '/analyze').length, before + 1, 'Comparison reuses the already analyzed role');
    assert.match(h.context.document.getElementById('compare-results').innerHTML, /Recurring essential gaps/);
    assert.match(h.context.document.getElementById('compare-results').innerHTML, /Docker \(2 roles\)/);

    const session = h.run('CHAT_SESSION_ID');
    await h.run('clearChat()');
    assert.ok(calls.some(c => c.route === `/chat/sessions/${session}` && c.method === 'DELETE'));
    assert.notEqual(h.run('CHAT_SESSION_ID'), session);
    fail = '/chat/documents/';
    await h.run("removeItem('jd', 'stored-a')");
    assert.equal(h.run("jds.some(j => j.id === 'stored-a')"), true, 'Failed cleanup must keep the visible document');
    fail = null;
    await h.run("removeItem('jd', 'stored-a')");
    assert.ok(calls.some(c => c.route.startsWith('/chat/documents/stored-a?session_id=') && c.method === 'DELETE'));
    assert.ok(calls.some(c => c.route === '/jds/stored-a' && c.method === 'DELETE'));
    assert.equal(h.run("jds.some(j => j.id === 'stored-a')"), false);

    h.run("resumes.push({id: 'my-resume', name: 'My resume', text: 'My uploaded resume text'}); currentResumeId = 'my-resume';");
    await h.run('loadSampleData()');
    assert.ok(calls.some(c => c.route === '/api/load-fixtures/sample' && c.method === 'POST'));
    assert.equal(h.run('currentResumeId'), 'my-resume', 'The sample loader keeps the selected personal resume');
    assert.equal(h.run("jds.filter(j => j.backend_id.startsWith('sample-')).length"), 5);
    assert.equal(h.run("jds.some(j => j.id === 'stored-b')"), true, 'The sample loader keeps other stored jobs');
    const loadedCount = h.run('jds.length');
    const resumeCount = h.run('resumes.length');
    await h.run('loadSampleData()');
    assert.equal(h.run('jds.length'), loadedCount, 'Reloading samples must not duplicate jobs');
    assert.equal(h.run('resumes.length'), resumeCount, 'Reloading samples must not duplicate resumes');
    assert.equal(calls.some(c => /load-fixtures\/(json|pdfs)/.test(c.route)), false, 'There is one sample route in the frontend');
    await h.run('resetSession()');
    assert.equal(h.run('resumes.length + jds.length'), 0);
    assert.ok(calls.some(c => c.route === '/jds' && c.method === 'DELETE'));
    console.log('HTML feature integration passed: settings, real rankings/filters, report download, comparison/cache, fixtures, and document/session cleanup.');
})().catch(error => { console.error(error); process.exitCode = 1; });
