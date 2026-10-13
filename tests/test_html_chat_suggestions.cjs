// Verify all demo suggestion buttons remain usable: node tests/test_html_chat_suggestions.cjs
const assert = require('node:assert/strict');
const { harness, response } = require('./test_html_chat_context.cjs');

(async () => {
    const h = harness();
    h.run("jds = [{...sampleJDs[0], id:'stored-python', backend_id:'stored-python'}]; selectedJdId='stored-python'; currentJdData={resume_id:currentResumeId,jd_id:selectedJdId,context_sufficient:true,match_score:100,skill_gap:{matched:[{skill_name:'Python'}],missing_essential:[],missing_preferred:[],extra:[]}}; updateChatContext(); openChatView();");
    const suggestions = Array.from(h.run('CHAT_SUGGESTIONS_GAP'));
    const buttons = () => h.elements.get('chat-suggestions').children;
    const calls = [];
    let finishAnswer;
    h.context.fetch = async (url, options) => {
        const body = JSON.parse(options.body);
        calls.push({url,body});
        if (url.endsWith('/chat/documents')) return response({context_id:'c'.repeat(64),chunk_count:3});
        if (url.endsWith('/chat')) return new Promise(resolve => { finishAnswer = () => resolve(response({answer:'Supported answer or clearly labelled advice',context_sufficient:true,citations:[]})); });
        throw new Error(`Unexpected ${url}`);
    };
    for(const question of suggestions) {
        assert.equal(buttons().length, 4);
        assert.equal(buttons().every(button => !button.disabled), true);
        const clicked = buttons().find(button => button.textContent === question);
        const pending = clicked.onclick();
        while(!finishAnswer) await new Promise(resolve => setImmediate(resolve));
        assert.equal(buttons().every(button => button.disabled), true, 'Prevent repeated clicks while awaiting an answer');
        assert.equal(calls.at(-1).body.message, question);
        assert.equal(calls.at(-1).body.context_id, 'c'.repeat(64));
        finishAnswer(); finishAnswer = null;
        await pending;
        assert.equal(buttons().length, 4, 'Keep all four buttons after every answer');
        assert.equal(buttons().every(button => !button.disabled), true);
    }
    assert.equal(calls.filter(c => c.url.endsWith('/chat')).length, 4);
    assert.equal(calls.filter(c => c.url.endsWith('/chat/documents')).length, 1, 'All suggestions share the completed report context');
    assert.equal(h.run('chatHistory.length'), 8);
    h.context.fetch = async () => ({ok:false,status:503,json:async()=>({detail:'Temporary provider error'})});
    await buttons()[0].onclick();
    assert.equal(buttons().length, 4);
    assert.equal(buttons().every(button => !button.disabled), true, 'Errors must leave the suggestions available for retry');
    console.log('All four suggestion buttons passed: correct questions/context, busy state, persistence beyond three answers, and retry.');
})().catch(error => { console.error(error); process.exitCode = 1; });
