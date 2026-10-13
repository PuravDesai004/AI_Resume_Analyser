// Offline screen-flow checks: node tests/test_html_placement_flow.cjs
const assert = require('node:assert/strict');
const { harness, response } = require('./test_html_chat_context.cjs');

(async () => {
    const h = harness();
    const active = id => h.elements.get(id)?.classList.contains('active') || false;
    h.run("updateChatContext(); switchView('placement')");
    assert.equal(active('view-setup'), true);
    assert.equal(active('view-matches'), false, 'No empty results screen before matching');
    assert.equal(active('view-gap'), false);
    assert.equal(h.elements.get('candidate-chip').hidden, true);
    h.run("switchView('matches')");
    assert.equal(active('view-setup'), true, 'Results require a completed ranking');
    h.run("switchView('gap')");
    assert.equal(active('view-gap'), false, 'A gap report requires completed analysis');

    h.run("jds = [{ id:'python-job', backend_id:'python-job', title:'Python role', company:'Example', location:'Remote', text:sampleJDs[0].text }];");
    h.context.fetch = async (url, options = {}) => {
        const body = options.body ? JSON.parse(options.body) : {};
        if(url.endsWith('/rank')) return response({results:[{jd_id:'python-job',title:'Python role',company:'Example',location:'Remote',rank:1,similarity_score:80,snippet:'Python required'}]});
        if(url.endsWith('/analyze')) return response({resume_id:body.resume_id,jd_id:body.jd_id,context_sufficient:true,match_score:80,summary:'Actual report',strengths:[],weaknesses:[],recommendations:[],requirement_match:[],skill_gap:{matched:[],missing_essential:['Docker'],missing_preferred:[],extra:[]}});
        throw new Error(`Unexpected ${url}`);
    };
    await h.run('runAnalysis()');
    assert.equal(active('view-setup'), false, 'Hide the input form after matching');
    assert.equal(active('view-matches'), true);
    assert.equal(h.elements.get('candidate-chip').hidden, false);
    assert.equal(h.run('lastRankings.length'), 1);
    await h.run("loadAnalysis(currentResumeId, 'python-job')");
    assert.equal(active('view-gap'), true);
    assert.equal(active('view-matches'), false);
    assert.equal(active('view-setup'), false);
    assert.equal(h.run('openChatView()'), true);
    h.run("switchView('matches')");
    assert.equal(active('view-matches'), true);
    assert.equal(h.run('lastRankings.length'), 1, 'Back to results retains the ranking');
    assert.equal(h.run('chatPanelOpen'), false);
    h.run("switchView('setup')");
    assert.equal(active('view-setup'), true);
    assert.equal(active('view-matches'), false);
    assert.equal(h.run('lastRankings.length'), 1, 'Returning to setup does not discard results');
    h.run("onCandidateSelected('res_devops_eng')");
    assert.equal(h.run('lastRankings.length'), 0, 'Changing the resume invalidates old results');
    h.run("switchView('matches')");
    assert.equal(active('view-setup'), true);

    h.context.fetch = async () => ({ok:false,status:503,json:async()=>({detail:'Provider unavailable'})});
    await h.run('runAnalysis()');
    assert.equal(active('view-setup'), true, 'Ranking failures stay on the input screen');
    assert.equal(active('view-matches'), false);
    assert.match(h.elements.get('placement-status').textContent,/Provider unavailable/);
    assert.equal(h.run('openChatView()'), false);
    console.log('Placement screen flow passed: setup, results gating, gap report, back paths and provider errors.');
})().catch(error => { console.error(error); process.exitCode = 1; });
