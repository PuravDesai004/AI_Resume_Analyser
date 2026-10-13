const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {harness, response} = require('./test_html_chat_context.cjs');

(async () => {
    const h = harness();
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'datasets', 'india_demo_100.json'), 'utf8'));
    const jobs = [{jd_id:'personal',title:'My job',company:'Mine',location:'Mumbai',full_text:'Personal job text'}];
    const calls = [];
    h.run("resumes.push({id:'personal-resume',name:'My resume',text:'Personal resume'}); currentResumeId='personal-resume';");
    h.context.fetch = async (url, options = {}) => {
        const route = url.replace('http://localhost:8010','');
        calls.push(route);
        if(route === '/api/load-fixtures/india') {
            const samples = data.jds.map(j=>({...j, backend_id:`saved-${j.id}`, id:`saved-${j.id}`}));
            for(const j of samples) if(!jobs.some(existing=>existing.jd_id===j.backend_id)) jobs.push({...j,jd_id:j.backend_id,full_text:j.text});
            return response({resumes:data.resumes,jds:samples,count:100,stored_count:jobs.length});
        }
        if(route === '/jds') return response({jds:jobs,count:jobs.length,cap:null});
        if(route === '/jds/count') return response({count:jobs.length,cap:null});
        if(route === '/jds/ids') return response({ids:jobs.map(j=>j.jd_id)});
        throw new Error(`Unexpected ${route}`);
    };
    h.context.document.getElementById('sample-dataset').value = 'india';
    await h.run('loadSampleData()');
    assert.ok(calls.includes('/api/load-fixtures/india'));
    assert.equal(h.run('resumes.length'),106);
    assert.equal(h.run('jds.length'),101);

    // A browser that did not press Load dataset still gets the demo profiles
    // associated with the shared backend, without losing its own selection.
    const second = harness();
    second.run("resumes.push({id:'personal-resume',name:'My resume',text:'Personal resume'}); currentResumeId='personal-resume';");
    second.context.fetch = async () => response({datasets:[{id:'india',resumes:data.resumes}]});
    await second.run('syncDemoResumes()');
    assert.equal(second.run('resumes.length'),106);
    assert.equal(second.run('currentResumeId'),'personal-resume');
    assert.equal(second.elements.get('resume-count').innerText,'106 resumes');
    await second.run('syncDemoResumes()');
    assert.equal(second.run('resumes.length'),106,'Repeated sync must not duplicate resumes');
    second.run("resumes = resumes.filter(r => r.id !== 'res_india_embedded_04'); saveResumeState();");
    const refreshed = harness({storage:second.storage});
    refreshed.context.fetch = second.context.fetch;
    await refreshed.run('syncDemoResumes()');
    assert.equal(refreshed.run('resumes.length'),105,'Refresh must respect deliberate deletions');
    assert.equal(refreshed.run('currentResumeId'),'personal-resume');
    const emptyBackend = harness();
    emptyBackend.context.fetch = async () => response({datasets:[]});
    await emptyBackend.run('syncDemoResumes()');
    assert.equal(emptyBackend.run('resumes.length'),5,'No dataset should be imported for an empty backend');
    assert.equal(h.run('currentResumeId'),'personal-resume');
    assert.equal(h.elements.get('jd-count').innerText,'101 jobs');
    assert.ok(!h.elements.get('store-summary').textContent.includes('null'));
    assert.equal(h.run('jdCap'),null);
    assert.match(h.elements.get('upload-notices').textContent,/All 100 demo jobs are saved/);
    await h.run('loadSampleData()');
    assert.equal(h.run('resumes.length'),106);
    assert.equal(h.run('jds.length'),101);

    h.context.rankCards = jobs.map((j,i)=>({...j,rank:i+1,embedding_similarity:80,similarity_score:80,
        score_source:i<25?'mercury':'similarity',fit_status:i<25?'assessed':'not_reviewed',
        model_fit:i<25?'Strong':null,model_fit_score:i<25?90:null}));
    h.run('lastRankings=rankCards;');
    h.run('renderRankings()');
    assert.equal(h.elements.get('role-list').children.length,101);
    assert.match(h.elements.get('role-list').children[25].innerHTML,/outside AI shortlist/);
    assert.ok(!h.elements.get('role-list').children[25].innerHTML.includes('Role fit ·'));
    h.context.document.getElementById('filter-score').value='50';
    h.run('renderRankings()');
    assert.equal(h.elements.get('role-list').children.length,25,'A role-fit filter must not treat unreviewed similarity as AI fit');
    console.log('India dataset HTML passed: 100 additions, personal data preserved, idempotent reload, uncapped counts, and honest unreviewed cards.');
})().catch(error=>{console.error(error);process.exitCode=1;});
