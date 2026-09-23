const {chromium}=require('C:/Users/Hello/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert');
(async()=>{
 const dir=path.resolve('result/opening_prediction/no_cold_start'),data=JSON.parse(fs.readFileSync(path.join(dir,'期限看板数据.json'),'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true}),context=await browser.newContext({offline:true,viewport:{width:1600,height:1100}}),page=await context.newPage(),errors=[],network=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
 await page.goto(require('url').pathToFileURL(path.join(dir,'无冷启动_3秒小组合.html')).href);
 await page.waitForSelector('body[data-ready="true"]');
 const pct=x=>x==null?'—':(100*x).toFixed(1)+'%';let cells=0;
 for(const segment of ['all','opening_0_9','later_10_59'])for(const model of ['selected','baseline'])for(const kind of ['cum','inc'])for(const sample of ['common_30','per_horizon'])for(const signal of ['all','0.2','0.4','0.6']){
  await page.evaluate(({segment,model,kind,sample,signal})=>{
   for(const [id,value] of Object.entries({segment,horizonModel:model,horizonKind:kind,horizonSample:sample,horizonSignal:signal}))document.getElementById(id).value=value;
   document.getElementById('segment').dispatchEvent(new Event('change'));
  },{segment,model,kind,sample,signal});
  const shown=await page.locator('[data-hw]').evaluateAll(es=>es.map(e=>({w:e.dataset.hw,u:+e.dataset.u,text:e.textContent,title:e.title})));
  assert.equal(shown.length,240);
  const rows=data.summary.filter(r=>r.segment===segment&&r.kind===kind&&r.sample===sample&&r.signal===signal);
  for(const c of shown){const id=c.w==='adaptive'?'adaptive_'+model:model+'_w'+c.w,r=rows.find(r=>r.model===id&&r.horizon===c.u);assert.equal(c.text,pct(r.accuracy));assert.equal(c.title,`${r.n}任务，${r.days}日`);cells++}
  await page.locator('[data-hw="3"][data-u="30"]').click();const r=rows.find(r=>r.model===model+'_w3'&&r.horizon===30),detail=await page.locator('#horizonDetail').textContent();
  assert(detail.includes('未来30秒'));assert(detail.includes(pct(r.accuracy)));assert(detail.includes(`${r.n}个有效任务／${r.days}日`));assert(detail.includes(pct(r.scheduled_coverage)));assert(detail.includes(pct(r.flat_share)));
  const series=rows.filter(r=>r.model===model+'_w3'&&r.accuracy!=null).sort((a,b)=>a.horizon-b.horizon);
  const points=await page.locator('#horizonChart circle').evaluateAll(es=>es.map(e=>({x:+e.getAttribute('cx'),y:+e.getAttribute('cy')})));
  assert.equal(points.length,series.length);points.forEach((p,i)=>{assert(Math.abs(p.x-(55+(series[i].horizon-1)*1015/29))<1e-9);assert(Math.abs(p.y-(235-210*series[i].accuracy))<1e-9)});
 }
 await page.selectOption('#segment','all');await page.selectOption('#horizonModel','selected');await page.selectOption('#horizonKind','cum');await page.selectOption('#horizonSample','common_30');await page.selectOption('#horizonSignal','all');await page.locator('[data-hw="3"][data-u="3"]').click();
 await page.locator('#horizonPanel').screenshot({path:path.join(dir,'网页验收_完整期限.png')});
 assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
 fs.writeFileSync(path.join(dir,'完整期限网页验收.json'),JSON.stringify({offline:true,metric_cells_checked:cells,errors,external_requests:network},null,2));await browser.close();console.log('Verified '+cells+' horizon cells, samples, details and curve coordinates offline.');
})().catch(e=>{console.error(e);process.exit(1)});
