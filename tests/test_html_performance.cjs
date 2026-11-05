// Run with: node tests/test_html_performance.cjs
const assert = require('node:assert/strict');
const { harness, response } = require('./test_html_chat_context.cjs');

(async () => {
    const h = harness();
    const pending = new Map();
    let nextTimer = 0;
    h.context.setTimeout = (callback, delay) => {
        assert.ok(delay > 0 && delay <= 500, 'Filtering remains responsive while keystrokes are coalesced');
        const id = ++nextTimer;
        pending.set(id, callback);
        return id;
    };
    h.context.clearTimeout = id => pending.delete(id);
    const calls = [];
    h.context.renderList = type => calls.push(`list:${type}`);
    h.context.renderRankings = () => calls.push('rankings');
    h.run('queueJobSearch(); queueJobSearch(); queueRankingFilter(); queueRankingFilter()');
    assert.equal(pending.size, 2, 'Each input group retains only its most recent pending update');
    for (const callback of pending.values()) callback();
    assert.deepEqual(calls, ['list:jd', 'rankings']);

    // A hidden tab must make no health requests, and visible polls must not overlap.
    h.context.setTimeout = () => 1;
    h.context.clearTimeout = () => {};
    h.context.document.getElementById('conn-dot').setAttribute = () => {};
    let fetches = 0, resolveHealth;
    h.context.fetch = () => { fetches++; return new Promise(resolve => { resolveHealth = resolve; }); };
    h.context.document.hidden = true;
    await h.run('pollServerConnection()');
    assert.equal(fetches, 0);
    h.context.document.hidden = false;
    const inFlight = h.run('pollServerConnection()');
    await h.run('pollServerConnection()');
    assert.equal(fetches, 1, 'A second poll must not start while a health check is pending');
    resolveHealth(response({ gemini_configured: true, model: 'test' }));
    await inFlight;
    h.context.fetch = async () => { fetches++; return response({ gemini_configured: true, model: 'test' }); };
    await h.run('pollServerConnection()');
    assert.equal(fetches, 2, 'Polling resumes after the previous check completes');
    console.log('HTML performance checks passed: debounced filters, hidden-tab polling, and non-overlapping health checks.');
})().catch(error => { console.error(error); process.exitCode = 1; });
