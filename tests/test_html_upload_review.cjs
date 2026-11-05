// Run with: node tests/test_html_upload_review.cjs
const assert = require('node:assert/strict');
const { harness, response } = require('./test_html_chat_context.cjs');

(async () => {
    const h = harness();
    const zone = h.context.document.getElementById('res-file-mode');
    zone.setAttribute = () => {};
    zone.removeAttribute = () => {};
    h.run("jds = [{id:'job', backend_id:'job', title:'Role', company:'Company', location:'Remote', text:'Job'}]; resumes = [{id:'previous', name:'Previous', text:'Previous resume'}]; currentResumeId = 'previous'; currentJdData = {resume_id:'previous', jd_id:'job', context_sufficient:true}");
    h.context.filesForTest = [{name:'first.txt',size:20}, {name:'second.txt',size:20}];
    let releaseFirst, calls = 0;
    h.context.fetch = () => {
        calls++;
        const doc = calls === 1
            ? {name:'first.txt',text:'First resume',pages:[{text:'First resume',page:1}],words:2,status:'Valid'}
            : {name:'second.txt',text:'Second resume',pages:[{text:'Second resume',page:1}],words:2,status:'Valid'};
        if (calls === 1) return new Promise(resolve => { releaseFirst = () => resolve(response(doc)); });
        return Promise.resolve(response(doc));
    };
    const upload = h.run("handleFileSelect('res', filesForTest)");
    assert.equal(h.elements.get('run-ranking-btn').disabled, true, 'Matching is disabled during an upload');
    assert.match(h.elements.get('placement-readiness').textContent, /upload/i);
    releaseFirst();
    await upload;
    assert.equal(calls, 2);
    assert.equal(h.run('resumes.length'), 3);
    assert.equal(h.run('resumes.find(r => r.id === currentResumeId).text'), 'Second resume');
    assert.equal(h.elements.get('run-ranking-btn').disabled, false, 'Matching is re-enabled after a valid batch');
    assert.equal(h.run('currentJdData'), null, 'A new selection cannot retain an old analysis');
    console.log('HTML upload checks passed: batch progress, latest resume selection, and analysis invalidation.');
})().catch(error => { console.error(error); process.exitCode = 1; });
