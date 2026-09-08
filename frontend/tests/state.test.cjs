const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function setup(fetch) {
    const nodes = new Map();
    class Element {
        constructor() {
            this.value = ''; this.content = ''; this.style = {};
            this.classList = {add() {}, remove() {}};
        }
        set innerHTML(value) { this.content = value; }
        get innerHTML() { return this.content; }
        set textContent(value) { this.content = value; }
        get textContent() { return this.content; }
        set innerText(value) { this.content = value; }
        appendChild() {} replaceChildren() { this.content = ''; }
    }
    const document = {
        getElementById(id) {
            assert.ok(html.includes(`id="${id}"`), `The actual HTML must contain ${id}`);
            if (!nodes.has(id)) nodes.set(id, new Element());
            return nodes.get(id);
        },
        querySelectorAll() { return []; }, querySelector() { return null; },
        createElement() { return new Element(); }, addEventListener() {},
    };
    const context = vm.createContext({document, fetch, Headers, AbortController, DOMException, FormData, alert() {}});
    vm.runInContext(script, context);
    vm.runInContext("documentId = 'document-one';", context);
    document.getElementById('qa-input').value = 'When is payment due?';
    return {context, document, run: code => vm.runInContext(code, context)};
}
function deferred() { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b;}); return {promise,resolve,reject}; }
function response(answer) { return {ok:true, json: async () => ({answer, sources:[]})}; }


test('new upload clears every previous document panel', () => {
    const app = setup();
    const ids = ['heatmap-container','summary-container','qa-output','negotiate-output',
        'obligations-container','compare-container','visuals-container','visual-analysis-output'];
    ids.forEach(id => app.document.getElementById(id).textContent = 'PRIVATE OLD DOCUMENT');
    app.run('resetDocumentViews()');
    ids.forEach(id => assert.equal(app.document.getElementById(id).textContent, 'Upload a contract to begin.'));
    assert.equal(app.run('documentId'), null);
});

test('late response from previous document cannot repopulate results', async () => {
    const network = deferred();
    const app = setup(() => network.promise);
    const pending = app.run('executeQA()');
    app.run('resetDocumentViews()');
    network.resolve(response('PRIVATE OLD DOCUMENT'));
    await pending;
    assert.equal(app.document.getElementById('qa-output').textContent, 'Upload a contract to begin.');
});

test('late network failure from previous document cannot replace new content', async () => {
    const network = deferred();
    const app = setup(() => network.promise);
    const pending = app.run('executeQA()');
    app.run('resetDocumentViews()');
    network.reject(new TypeError('Old network failed'));
    await pending;
    assert.equal(app.document.getElementById('qa-output').textContent, 'Upload a contract to begin.');
});

test('document change during JSON decoding is also discarded', async () => {
    const body = deferred();
    const app = setup(async () => ({ok:true, json:() => body.promise}));
    const pending = app.run('executeQA()');
    await new Promise(resolve => setImmediate(resolve));
    app.run('resetDocumentViews()');
    body.resolve({answer:'OLD RESULT',sources:[]});
    await pending;
    assert.equal(app.document.getElementById('qa-output').textContent, 'Upload a contract to begin.');
});

test('most recent question wins when requests finish out of order', async () => {
    const first = deferred(), second = deferred(); let calls=0;
    const app = setup(() => ++calls === 1 ? first.promise : second.promise);
    const oldRequest = app.run('executeQA()');
    const newRequest = app.run('executeQA()');
    second.resolve(response('LATEST ANSWER')); await newRequest;
    first.resolve(response('OLD ANSWER')); await oldRequest;
    const result = app.document.getElementById('qa-output').innerHTML;
    assert.ok(result.includes('LATEST ANSWER'));
    assert.ok(!result.includes('OLD ANSWER'));
});

test('document API calls identify their source upload', async () => {
    let headers;
    const app = setup(async (url, options) => {headers=options.headers;return response('OK');});
    await app.run('executeQA()');
    assert.equal(headers.get('X-Document-Id'), 'document-one');
});
