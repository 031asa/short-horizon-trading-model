const {chromium}=require('C:/Users/Hello/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert');
(async()=>{const dir=path.resolve('result/opening_prediction/no_cold_start'),data=JSON.parse(fs.readFileSync(path.join(dir,'看板数据.json'),'utf8'));
const browser=await chromium.launch({channel:'msedge',headless:true}),context=await browser.newContext({offline:true,viewport:{width:1500,height:1100}}),page=await context.newPage(),errors=[],network=[];
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
await page.goto(require('url').pathToFileURL(path.join(dir,'无冷启动_3秒小组合.html')).href);await page.waitForSelector('body[data-ready="true"]');const pct=x=>(100*x).toFixed(1)+'%';let cells=0;
for(const sample of ['common','own'])for(const segment of ['all','opening_0_9','later_10_59']){
 await page.selectOption('#sample',sample);await page.selectOption('#segment',segment);
 for(const w of data.rules.windows){const tr=page.locator(`tr[data-w="${w}"]`),actual=await tr.locator('td').allTextContents(),b=data.summary.find(r=>r.sample===sample&&r.segment===segment&&r.model==='baseline_w'+w),s=data.summary.find(r=>r.sample===sample&&r.segment===segment&&r.model==='selected_w'+w);
  assert.equal(actual[1],String(s.n));assert.equal(actual[2],pct(b.accuracy));assert.equal(actual[3],pct(s.accuracy));assert.equal(actual[4],b.logloss.toFixed(3));assert.equal(actual[5],s.logloss.toFixed(3));cells+=5;
  await tr.click();assert.equal(await page.locator('#factors tr').count(),4);
  for(const mode of ['fixed_rank','frozen_threshold']){await page.selectOption('#coverageMode',mode);let i=0;
   for(const level of data.rules.coverage_levels){const shown=await page.locator('#confidence tr').nth(i++).locator('td').allTextContents(),b=data.confidence.find(r=>r.sample===sample&&r.mode===mode&&r.level===level&&r.model==='baseline_w'+w),s=data.confidence.find(r=>r.sample===sample&&r.mode===mode&&r.level===level&&r.model==='selected_w'+w);assert.equal(shown[1],pct(b.accuracy));assert.equal(shown[4],pct(s.accuracy));assert.equal(shown[5],pct(s.coverage));cells+=3;}
  }
 }
 assert.equal(await page.locator('#chart rect').count(),14);assert.equal(await page.locator('#choice tr').count(),4);
}
await page.selectOption('#sample','common');await page.selectOption('#segment','all');await page.click('tr[data-w="3"]');await page.selectOption('#coverageMode','frozen_threshold');await page.screenshot({path:path.join(dir,'网页验收_无冷启动总览.png'),fullPage:true});await page.locator('#detail').screenshot({path:path.join(dir,'网页验收_无冷启动组合.png')});assert.deepEqual(errors,[]);assert.deepEqual(network,[]);fs.writeFileSync(path.join(dir,'网页验收.json'),JSON.stringify({offline:true,metric_cells_checked:cells,errors,external_requests:network},null,2));await browser.close();console.log('Checked '+cells+' data cells offline.');})().catch(e=>{console.error(e);process.exit(1)});
