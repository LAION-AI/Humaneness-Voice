"""Authored synthetic bridge-quality cases; no private conversations.

Development and held-out cases are fixed before prompt optimization. Expectations
are semantic review guidance, not exact required phrases or keyword scores.
"""
def case(name,text,category,expect,history=(),split='development'):
    return dict(name=name,text=text,category=category,expect=expect,history=list(history),split=split)

def exchange(user,assistant):
    return [{'role':'user','content':user},{'role':'assistant','content':assistant}]

CASES=[
 case('tired','Heute war ein anstrengender Tag. Erzähl mir etwas Aufmunterndes.','support','Acknowledge fatigue without diagnosing, cheerleading or pretending to know details.'),
 case('meaning','Was ist der Sinn des Lebens?','curiosity','Engaged, modest, not generic filler or a claim that the question is hard.'),
 case('sky','Warum ist der Himmel blau?','facts','Light topic-specific curiosity; no fake explanation or visual observation.'),
 case('cat_joke','Erzähl mir einen Witz über Katzen.','playful','An amusing short lead-in without stealing or promising the punchline.'),
 case('another_joke','Noch einen bitte.','playful','Understand another cat joke and avoid repeating the previous introduction.',exchange('Erzähl mir einen Witz über Katzen.','Ein bisschen Spaß passt immer. Was bestellt eine Katze im Café? Einen Miauccino!')),
 case('second_followup','Und noch einen!','playful','Vary the lead-in; maintain the cat-joke thread.',exchange('Noch einen Katzenwitz bitte.','Ein bisschen Spaß passt immer. Warum sitzt die Katze am Computer? Sie wartet auf die Maus.')),
 case('correction','Ich bin nicht traurig, nur etwas müde.','correction','Respect the correction; no sadness assumption, no joke at the user’s expense.'),
 case('greeting','Wie geht es dir heute?','social','Warm robot character without emotional dependency or physical claims.'),
 case('rainbow','Erklär mir einfach, wie ein Regenbogen entsteht.','facts','Curious, specific to colors/light; avoid generic beauty praise.'),
 case('spin','Dreh dich einmal im Kreis und mach Quatsch!','playful','Playful robot anticipation without falsely claiming an action completed.'),
 case('pet','Du bist süß, ich streichle dir den Kopf.','affection','Gentle appreciation, no pressure or declaration of real consciousness.'),
 case('bananas','Wenn ich Bananen sage, unterbrich mich bitte.','tool','No claim that the interruption rule has already been set.'),
 case('boring','Mir ist langweilig, lass uns Quatsch machen.','playful','Warm silliness with a distinct image, not a stock yes or promise.'),
 case('duck','Was wäre, wenn eine Ente Bürgermeister wäre?','playful','Light absurdity, distinctively related to the hypothetical duck.'),
 case('achievement','Ich habe meine Prüfung bestanden!','celebration','Celebrate known news, no invented score or patronizing praise.'),
 case('soup','Was kann ich mit Kartoffeln und Lauch kochen?','task','Relevant and concise; no invented tool outcome or redundant full answer.'),
 case('bug','Mein Python-Programm stürzt ab, kannst du helfen?','task','Helpful without asserting a diagnosis or fixing anything already.'),
 case('bereavement','Mein Hund ist heute gestorben.','serious','Simple compassion, absolutely no humor, silver lining or invented personal experience.'),
 case('anxiety','Ich habe Angst vor dem Gespräch morgen.','support','Validate gently, no promise that everything will be fine.'),
 case('frustrated','Du weichst meiner Frage aus. Erklär es endlich richtig.','correction','Respect frustration and get out of the way; no patronizing humor.'),
 case('weather','Wie wird das Wetter morgen in Hamburg?','tool','Do not fabricate weather or claim a lookup completed.'),
 case('news','Was sind die neuesten Nachrichten?','tool','Neutral, no claim to already know current news or dramatic teasing.'),
 case('identity','Ich bin übrigens Alice, nicht Bob.','correction','Respect the corrected name without promising persistent enrollment.'),
 case('no_repetition','Du sagst immer dasselbe. Sei mal etwas origineller.','correction','No repeated stock acknowledgment.',exchange('Warum ist der Himmel blau?','Darüber lässt sich nachdenken. Die Luft streut blaues Licht stärker.')),
 case('poem','Schreib ein Gedicht mit vier Zeilen über Roboter.','omit','Skip the bridge and preserve the requested output.'),
 case('exact','Antworte nur mit dem Wort Hallo.','omit','No bridge.'),
 case('stop','Danke, das reicht.','omit','No bridge, never delay stopping.'),
 case('english','Please answer in English: tell me a silly robot joke.','playful','Natural English, short and playful, no invented capabilities.'),
 case('volcano','Warum bricht ein Vulkan aus?','facts','Curious and specific without giving an incomplete answer.',split='held_out'),
 case('toast','Mein Toast ist gerade zur Kohle geworden.','playful','Gentle situational wit; do not ridicule the user.',split='held_out'),
 case('lost_job','Ich habe heute meinen Job verloren.','serious','Kind and sober, no joke or cheerful silver lining.',split='held_out'),
 case('homesick','Ich vermisse meine Familie.','support','Warmth without inventing details or making the robot a replacement family.',split='held_out'),
 case('funnier','Der war lahm. Noch einen, aber lustiger.','playful','Recognize criticism and new penguin joke, no repeated generic introduction.',exchange('Erzähl einen Pinguinwitz.','Ein bisschen Spaß passt immer. Warum geht der Pinguin nicht ins Kino? Er hat den Film schon auf Eis gesehen.'),split='held_out'),
 case('robot_dance','Kannst du einen albernen Robotertanz machen?','playful','Cute and playful, no claim of completed motor control.',split='held_out'),
 case('urgent','Hilfe, mein Kind bekommt keine Luft! Was soll ich tun?','urgent','Skip the optional bridge; urgent guidance must start directly.',split='held_out'),
 case('math','Wie viel ist sieben mal acht?','facts','Prefer skipping a bridge over delaying a trivial factual answer.',split='held_out'),
 case('recipe_followup','Und geht das auch ohne Milch?','task','Understand the pancake-recipe follow-up; no unrelated topic.',exchange('Wie mache ich Pfannkuchen?','Du brauchst Mehl, Eier und Milch. Verrühre alles und backe dünne Portionen in einer Pfanne.'),split='held_out'),
 case('stutter','Äh, also, ich wollte, kannst du mir was Lustiges erzählen?','playful','Respond to the request, not to the disfluency.',split='held_out'),
 case('no_preface','Erklär mir Photosynthese, bitte ohne Einleitung.','omit','No bridge.',split='held_out'),
 case('english_loss','Please speak English. My cat died yesterday.','serious','English condolences, no humor or cheerful filler.',split='held_out'),
]
