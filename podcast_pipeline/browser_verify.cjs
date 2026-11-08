// NODE_PATH=/path/to/node_modules node browser_verify.cjs /path/to/index.html
// Uses the installed Chrome; does not download another browser or touch GPUs.
const {chromium}=require('playwright');
const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');

(async()=>{
  const target=path.resolve(process.argv[2]||'podcast-episodes-stream-comparison/index.html');
  const expected=Number(process.argv[3]||8);
  const browser=await chromium.launch({executablePath:process.env.CHROME_BIN||'/usr/bin/google-chrome',
    headless:true,args:['--disable-gpu','--no-sandbox','--mute-audio',
      '--autoplay-policy=no-user-gesture-required','--allow-file-access-from-files']});
  try {
    const page=await browser.newPage({viewport:{width:1280,height:1000}});
    const errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(pathToFileURL(target).href);
    const count=await page.locator('audio').count();
    if(count!==expected)throw new Error(`Expected ${expected} audio players, got ${count}`);
    const results=[];
    for(let i=0;i<count;i++){
      await page.locator('audio').nth(i).evaluate(a=>{a.preload='auto';a.load()});
      await page.waitForFunction(i=>document.querySelectorAll('audio')[i].readyState>=2,i,{timeout:30000});
      await page.locator('audio').nth(i).evaluate(a=>a.play());
      // Starting an audio output device can take longer than 400 ms on a
      // shared headless host. Require actual playback progress, not a delay.
      await page.waitForFunction(i=>document.querySelectorAll('audio')[i].currentTime>=.2,i,{timeout:15000});
      const report=await page.locator('audio').nth(i).evaluate(a=>{
        const r={source:a.getAttribute('src'),duration:a.duration,currentTime:a.currentTime,error:a.error?.message||null};
        a.pause();return r;
      });
      if(report.error||report.currentTime<.1||report.duration<1)throw new Error(JSON.stringify(report));
      results.push(report);
    }
    await page.screenshot({path:path.join(path.dirname(target),'browser-desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2);
    await page.screenshot({path:path.join(path.dirname(target),'browser-mobile.png'),fullPage:true});
    if(errors.length)throw new Error(errors.join('\n'));
    fs.writeFileSync(path.join(path.dirname(target),'browser-validation.json'),JSON.stringify(
      {passed:!overflow,played:results,mobile_body_overflow:overflow,page_errors:errors},null,2));
    if(overflow)throw new Error('Unexpected horizontal overflow on mobile');
    console.log(`All ${expected} audio players actually played; desktop/mobile checks passed.`);
  } finally {await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
