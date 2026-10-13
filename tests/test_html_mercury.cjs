// Same placement screens; typed ranking metadata and explicit key preservation.
const assert = require('node:assert/strict');
const { harness, response } = require('./test_html_chat_context.cjs');

(async () => {
    const h = harness();
    const calls = [];
    let note = 'Ranked by Mercury role fit';
    let results = [
        { jd_id: 'b', rank: 1, title: 'React role', company: 'Beta', location: 'Mumbai (Remote)', snippet: 'React', similarity_score: 68, embedding_similarity: 60, preference_boost: 8, baseline_rank: 2, model_fit: 'Strong', model_fit_score: 72, score_source: 'mercury', ranking_score: 80, location_bonus: 5, arrangement_bonus: 3, location_distance_km: 0, location_reason: 'Preferred city: Mumbai', decision_confidence: .8 },
        { jd_id: 'a', rank: 2, title: 'ML role', company: 'Alpha', location: 'Ahmedabad', snippet: 'ML', similarity_score: 85, embedding_similarity: 85, baseline_rank: 1, model_fit: 'Moderate', model_fit_score: 75, score_source: 'mercury', ranking_score: 75, location_bonus: 0, location_distance_km: 440.6, location_reason: 'Beyond 300 km of Mumbai', decision_confidence: .7 },
    ];
    h.context.fetch = async (url, options = {}) => {
        const route = url.replace('http://localhost:8010', '');
        const body = options.body ? JSON.parse(options.body) : undefined;
        calls.push({ route, body });
        if (route === '/api/config') return response({ generation_model: 'gemini-model', embedding_model: 'embedding', jd_cap: 15, jd_count: 2, openrouter_configured: true, openrouter_masked: 'masked', decision_model: 'inception/mercury-decide:free', decision_enabled: true });
        if (route === '/rank') return response({ results, ranking_method: 'mercury', ranking_note: note });
        if (route.startsWith('/chat/') && options.method === 'DELETE') return response({ deleted_chunks: 0 });
        throw new Error(`Unexpected call ${route}`);
    };
    await h.run('loadSettings()');
    assert.match(h.elements.get('openrouter-key-status').textContent, /Configured/);
    assert.equal(h.context.document.getElementById('settings-openrouter-key').value, '', 'The saved key is never sent to the HTML');
    await h.run('saveSettings()');
    let saved = calls.filter(c => c.route === '/api/config' && c.body).at(-1).body;
    assert.equal('openrouter_api_key' in saved, false, 'Blank input preserves the saved key');
    h.context.document.getElementById('settings-openrouter-key').value = 'fake-key-for-test';
    await h.run('saveSettings()');
    saved = calls.filter(c => c.route === '/api/config' && c.body).at(-1).body;
    assert.equal(saved.openrouter_api_key, 'fake-key-for-test');
    assert.equal(h.context.document.getElementById('settings-openrouter-key').value, '');
    h.elements.get('remove-openrouter-key').checked = true;
    await h.run('saveSettings()');
    saved = calls.filter(c => c.route === '/api/config' && c.body).at(-1).body;
    assert.equal(saved.openrouter_api_key, '');
    h.context.document.getElementById('rank-sort').value = 'score';
    await h.run("generateRankings('res_python_dev', true)");
    let cards = h.elements.get('role-list').children;
    assert.match(cards[0].innerHTML, /React role/);
    assert.match(cards[0].innerHTML, /<small>Role fit · Strong<\/small>/);
    assert.equal((cards[0].innerHTML.match(/Role fit · Strong/g) || []).length, 1);
    assert.ok(!cards[0].innerHTML.includes('model-fit') && !cards[0].innerHTML.includes('Model fit:'), 'The duplicate category badge is removed');
    assert.match(cards[0].innerHTML, /<strong>72\/100<\/strong>/);
    assert.ok(!cards[0].innerHTML.includes('Mercury fit'), 'The user-facing score has a model-independent name');
    assert.match(cards[0].innerHTML, /Location \+5/);
    assert.match(cards[0].innerHTML, /Work arrangement \+3/);
    assert.match(cards[0].innerHTML, /Ranking total: 80 points/);
    assert.match(cards[1].innerHTML, /Location \+0 · ~440.6 km · Beyond 300/);
    assert.equal(h.elements.get('minimum-score-label').textContent, 'Minimum role fit');
    assert.match(h.elements.get('ranking-status').textContent, /Mercury/);
    h.context.document.getElementById('rank-sort').value = 'similarity';
    h.run('renderRankings()');
    assert.match(h.elements.get('role-list').children[0].innerHTML, /ML role/);
    h.context.document.getElementById('filter-score').value = '73';
    h.run('renderRankings()');
    assert.equal(h.elements.get('role-list').children.length, 1, 'Filter uses role fit rather than similarity or the ranking bonus');
    assert.match(h.elements.get('role-list').children[0].innerHTML, /ML role/);
    h.context.document.getElementById('filter-score').value = '0';
    results[0].model_fit_score = 0;
    await h.run("generateRankings('res_python_dev', true)");
    assert.match(h.elements.get('role-list').children[1].innerHTML, /<strong>0\/100<\/strong>/, 'A zero model fit must not become a similarity score');
    results = results.map(({ model_fit, model_fit_score, decision_confidence, ...card }) => ({...card, score_source: 'similarity', ranking_score: card.similarity_score}));
    note = 'Ranked by similarity. Add an OpenRouter key in Settings to enable Mercury.';
    await h.run("generateRankings('res_python_dev', true)");
    assert.ok(h.elements.get('role-list').children.every(c => !c.innerHTML.includes('Role fit ·') && !c.innerHTML.includes('Model fit:')), 'Fallback has no fabricated fit category');
    assert.match(h.elements.get('role-list').children[0].innerHTML, /<strong>85%<\/strong>/);
    assert.equal(h.elements.get('minimum-score-label').textContent, 'Minimum similarity');
    assert.match(h.elements.get('ranking-status').textContent, /Add an OpenRouter key/);
    console.log('HTML Mercury checks passed: main fit score, separate distance/location/work bonuses, ranking total, fit filtering, similarity sorting and honest fallback.');
})().catch(error => { console.error(error); process.exitCode = 1; });
