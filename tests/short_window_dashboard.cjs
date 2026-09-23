const {chromium}=require('C:/Users/Hello/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
(async()=>{
 const dir=path.resolve('result/opening_execution'),file=path.join(dir,'短观察期因子初筛.html'),data=JSON.parse(fs.readFileSync(path.join(dir,'短观察期_看板数据.json'),'utf8'));
 const curves=new Map(data.curves.map(r=>[[r.mode,r.expression,r.w,r.anchor,r.label_type,r.pair_set,r.horizon].join('|'),r]));
 const browser=await chromium.launch({channel:'msedge',headless:true}),context=await browser.newContext({offline:true,viewport:{width:1500,height:1100}}),page=await context.newPage();const errors=[],external=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))external.push(r.url())});
 await page.goto(require('url').pathToFileURL(file).href);await page.waitForSelector('body[data-ready="true"]');
 assert.equal(await page.locator('#index tr').count(),28);assert((await page.locator('#clockNote').innerText()).includes('t = T + W'));
 let cells=0;const fmt=v=>v==null?'—':(v>0?'+':'')+v.toFixed(3);
 for(const expression of ['F03_OFI','F03_OFI_latest']){
  await page.click('tr[data-factor="'+expression+'"]');
  for(const mode of ['wait','fixed'])for(const anchor of ['decision','arrival'])for(const pairs of ['own','diagonal_common'])for(const metric of ['rank_ic','ic'])for(const kind of ['cumulative','incremental']){
   await page.selectOption('#mode',mode);await page.selectOption('#anchor',anchor);await page.selectOption('#pairs',pairs);await page.selectOption('#metric',metric);await page.selectOption('#tableKind',kind);
   const actual=await page.locator('#heat td[data-u]').allTextContents(),expected=[];
   for(let w=1;w<=5;w++)for(let u=1;u<=30;u++)expected.push(fmt(curves.get([mode,expression,w,anchor,kind,pairs,u].join('|'))['mean_'+metric]));
   assert.deepEqual(actual,expected);cells+=actual.length;
  }
 }
 await page.selectOption('#mode','wait');await page.selectOption('#anchor','decision');await page.selectOption('#pairs','own');await page.selectOption('#metric','rank_ic');await page.selectOption('#tableKind','cumulative');
 await page.selectOption('#scope','candidate');assert.equal(await page.locator('#index tr').count(),11);assert.equal(await page.locator('tr[data-factor="F07_QuotePosition"]').count(),1);
 await page.selectOption('#scope','compressed');assert.equal(await page.locator('#index tr').count(),0);await page.selectOption('#scope','all');
 await page.click('tr[data-factor="F03_OFI"]');assert.equal(await page.locator('#cumulative circle').count(),150);assert.equal(await page.locator('#incremental circle').count(),150);
 assert.equal(await page.locator('#counts tr[data-w="1"] td').nth(1).innerText(),'3');assert.equal(await page.locator('#counts tr[data-w="5"] td').nth(1).innerText(),'11');
 await page.locator('#factorPanel').screenshot({path:path.join(dir,'网页验收_短观察期曲线.png')});
 for(const w of ['1','4']){
  await page.selectOption('#shortW',w);const rows=data.paired.filter(r=>r.mode==='wait'&&r.expression==='F03_OFI'&&r.w===+w&&r.anchor==='decision'&&r.label_type==='cumulative').sort((a,b)=>a.horizon-b.horizon);
  const shown=await page.locator('#diffTable tr[data-field="aligned_difference"] td').allTextContents();assert.deepEqual(shown.slice(1),rows.map(r=>fmt(r.aligned_difference)));
 }
 await page.click('tr[data-factor="A04_QISlope"]');assert.equal(await page.locator('#heat tr[data-w="1"] td[data-u="1"]').innerText(),'—');
 await page.selectOption('#shortW','2');assert((await page.locator('#diffTable').innerText()).includes('+'));
 await page.selectOption('#pairs','diagonal_common');assert.equal(await page.locator('#cumulative circle').count(),0);
 await page.selectOption('#pairs','own');await page.click('tr[data-factor="F03_OFI"]');await page.locator('#index').screenshot({path:path.join(dir,'网页验收_短观察期候选.png')});
 assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
 const result={html_sha256:crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),offline:true,browser_errors:errors,external_requests:external.length,table_cells_verified:cells,candidate_rows:11,compression_supported_rows:0,curve_points:300,pairwise_cells_verified:60,counts_verified:true,missing_slope_and_pairwise_behavior_verified:true};
 fs.writeFileSync(path.join(dir,'short_window_dashboard_verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
