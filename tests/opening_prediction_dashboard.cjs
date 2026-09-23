const {chromium}=require('C:/Users/Hello/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert');
(async()=>{
const dir=path.resolve('result/opening_prediction'),data=JSON.parse(fs.readFileSync(path.join(dir,'看板数据.json'),'utf8'));
const browser=await chromium.launch({channel:'msedge',headless:true}),context=await browser.newContext({offline:true,viewport:{width:1500,height:1100}}),page=await context.newPage();
const errors=[],network=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
await page.goto(require('url').pathToFileURL(path.join(dir,'短窗口涨跌预测检验.html')).href);await page.waitForSelector('body[data-ready="true"]');
let cells=0;const pct=x=>(100*x).toFixed(1)+'%';
for(const h of [1,2,3])for(const sample of ['common_windows','own_window']){
 await page.selectOption('#h',String(h));await page.selectOption('#sample',sample);
 assert.equal(await page.locator('#results tr').count(),13);
 for(const r of data.summary.filter(r=>r.horizon===h&&r.sample===sample)){
  const tr=page.locator(`#results tr[data-model="${r.model}"]`),b=data.baseline.find(b=>b.model===r.model&&b.horizon===h&&b.sample===sample);
  const actual=await tr.locator('td').allTextContents();
  assert.equal(actual[2],r.n+' / '+pct(r.coverage));assert.equal(actual[3],pct(r.accuracy));assert.equal(actual[4],pct(b.accuracy));assert.equal(actual[6],r.logloss.toFixed(3));cells+=4;
  await tr.click();const cm=data.confusion.find(c=>c.model===r.model&&c.horizon===h&&c.sample===sample).matrix;
  assert.deepEqual((await page.locator('#cm tr').allTextContents()).length,3);
  for(let i=0;i<3;i++)assert.deepEqual((await page.locator('#cm tr').nth(i).locator('td').allTextContents()).slice(1),cm[i].map(String));
  assert.equal(await page.locator('#dailyChart circle').count(),18);assert.equal(await page.locator('#dailyChart polyline').count(),2);
  assert.equal(await page.locator('#confidence tr').count(),5);
  const conf=data.confidence.find(c=>c.model===r.model&&c.horizon===h&&c.sample===sample&&c.threshold===.6);
  assert.equal(await page.locator('#confidence tr').nth(1).locator('td').nth(1).innerText(),String(conf.signals));
 }
}
await page.selectOption('#h','1');await page.selectOption('#sample','common_windows');await page.locator('#results tr[data-model="snapshot"]').click();
await page.screenshot({path:path.join(dir,'网页验收_预测总览.png'),fullPage:true});
await page.locator('#detail').screenshot({path:path.join(dir,'网页验收_预测明细.png')});
assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
fs.writeFileSync(path.join(dir,'网页验收.json'),JSON.stringify({offline:true,metric_cells_checked:cells,model_details_checked:78,errors,external_requests:network},null,2));
await browser.close();console.log('Offline dashboard verified: '+cells+' metric cells, 78 model details.');
})().catch(e=>{console.error(e);process.exit(1)});
