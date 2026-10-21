"""HTTPS typed-turn timing and actual browser scheduling; not physical audio."""
import argparse, asyncio, json, time
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--count',type=int,default=3);args=parser.parse_args()
    rows=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
        await page.goto('https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
        await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=90000)
        await page.evaluate('''()=>{window.probe={};const play=robo.play.bind(robo),message=robo.on.message;
          robo.play=raw=>{const id=new DataView(raw).getInt32(0,true);if(id){probe.chunks??=[];probe.chunks.push({at:performance.now(),seconds:(raw.byteLength-4)/2/24000,gap_ms:probe.chunks.length?Math.max(0,robo.ctx.currentTime-robo.next)*1000:0});}play(raw);};
          robo.on.message=m=>{if(m.type==='reply'){probe.reply_at=performance.now();probe.timing=m.timing;probe.text=m.text;}if(m.type==='turn_done')probe.done=true;if(m.type==='error')probe.error=m;message(m);};}''')
        try:
            for i in range(args.count):
                await page.evaluate('()=>{window.probe={start:performance.now(),chunks:[]};}')
                await page.locator('#message').fill('Bitte lies genau diese zwei Sätze vor: Heute machen wir einen kleinen Spaziergang durch den Park. Danach setzen wir uns auf eine Bank und genießen ganz in Ruhe die warme Nachmittagssonne.')
                await page.locator('#form').evaluate('f=>f.requestSubmit()')
                await page.wait_for_function('probe.done||probe.error',timeout=90000)
                row=await page.evaluate('probe');rows.append(row)
                Path(args.output).write_text(json.dumps(dict(physical_audio_device=False,rows=rows,page_errors=errors),indent=2))
                print(json.dumps(row),flush=True)
                await page.wait_for_function('!robo.ctx || robo.next<=robo.ctx.currentTime+.1',timeout=60000)
        finally:await page.evaluate('window.robo?.stop()');await browser.close()

if __name__=='__main__':asyncio.run(main())
