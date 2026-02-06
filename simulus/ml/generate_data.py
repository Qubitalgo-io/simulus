from __future__ import annotations

import json
import random
import re
from pathlib import Path


# templates are organized in 3 tiers per domain:
#   clear   -- unambiguous domain signal
#   subtle  -- domain implied but not stated directly
#   messy   -- colloquial, multi-sentence, or cross-domain bleed

DOMAIN_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "career": {
        "clear": [
            "I want to ask my boss for a raise",
            "I am thinking about quitting my job",
            "Should I accept the new job offer",
            "My manager is pushing me to relocate for work",
            "I got fired and need to figure out my next move",
            "I want to start my own business",
            "I am considering a career change to {field}",
            "My colleague is taking credit for my work",
            "I have a job interview at {company} tomorrow",
            "I want to negotiate a higher salary",
            "Should I report my boss for misconduct",
            "I am burning out at my current job",
            "My startup is running out of money",
            "I want to become a {profession} but I have no experience",
            "I was offered a promotion but it means more travel",
            "My coworker got the promotion I deserved",
            "I need to fire someone on my team",
            "The company is doing layoffs and I might be next",
            "Should I go back to my old job",
            "I want to freelance but I am scared of losing stability",
            "My boss wants me to do something unethical",
            "I am overqualified for my current role",
            "Should I take a pay cut to work somewhere I love",
            "I just got a terrible performance review",
            "I want to ask for a transfer to another department",
        ],
        "subtle": [
            "Everyone at the office is getting ahead except me",
            "My nine to five is killing me slowly",
            "I have been working 80 hour weeks and I do not see the point anymore",
            "The new management is toxic and people are leaving",
            "I keep getting passed over and I do not know why",
            "The industry is changing and my skills feel obsolete",
            "I have an idea that could be a real company but I do not know where to start",
            "My mentor retired and now I have no guidance at work",
            "I keep dreaming about doing something completely different with my life professionally",
            "The annual review cycle is coming and I am dreading it",
        ],
        "messy": [
            "so basically my boss called me into his office today and said they are restructuring. I have no idea what is going to happen to my role",
            "I have been at this company for 8 years and honestly I think I have peaked. there is nowhere to go and I am only {age}",
            "my friend just started a startup and wants me to join but I have a mortgage and two kids. is it crazy to even consider this",
            "I got two offers. one pays more but the other has better people and work-life balance. I do not know what matters more to me",
            "the job market is terrible right now but I cannot stand my manager for one more day. do I just quit with nothing lined up",
            "I built this team from scratch and now they want to replace me with someone from outside. I am devastated",
            "my side project is making almost as much as my salary. do I go all in or keep playing it safe",
            "HR just announced mandatory return to office. I moved 3 hours away during covid. what do I do",
        ],
    },
    "relationship": {
        "clear": [
            "Should I break up with my partner",
            "I think my spouse is hiding something from me",
            "My best friend betrayed my trust",
            "I want to propose to my girlfriend",
            "My parents disapprove of my relationship",
            "I found out my partner has been lying to me",
            "Should I forgive my friend for what they did",
            "I am falling for someone who is already taken",
            "My partner wants kids but I do not",
            "I am in a long-distance relationship and it is hard",
            "My ex wants to get back together",
            "I had a huge fight with my {relative}",
            "My partner and I disagree about everything lately",
            "I think my marriage is falling apart",
            "Should I tell my friend I have feelings for them",
            "My family is pressuring me to get married",
            "I caught my partner texting someone else",
            "My {relative} is toxic but I feel guilty cutting them off",
            "I want to reconnect with my estranged {relative}",
            "Should I move in with my partner",
            "My friend group is falling apart",
            "I do not trust my partner anymore",
            "My partner wants an open relationship",
            "I am jealous of my best friend's success",
            "Should I confront my friend about their behavior",
        ],
        "subtle": [
            "We used to talk every day but now it is just silence",
            "I feel like I am the only one trying in this thing",
            "She said she needs space and I do not know what that means",
            "My family is amazing but they are suffocating me",
            "He changed after the baby and I do not recognize him anymore",
            "I keep attracting the same kind of person and it always ends badly",
            "Everyone around me is getting married and I am still alone",
            "I said something I cannot take back and now there is this wall between us",
            "My parents are getting divorced and I am stuck in the middle",
            "I love them but I do not think I am in love anymore",
        ],
        "messy": [
            "so my partner of 5 years just told me they want to see other people. I am sitting here at 2am and I cannot breathe",
            "my mom and my wife do not get along and it is tearing me apart. every family gathering is a disaster",
            "we have been fighting about the same thing for 3 years. nothing changes. I am so tired",
            "I found messages on his phone. he says they are just friends but the messages say otherwise. what do I do",
            "my best friend since childhood and I had a falling out over money. it has been 6 months and I miss her but I am also still angry",
            "my sister is in a bad relationship and refuses to leave. I have tried everything but she keeps going back",
            "I am 35 and my parents still control every aspect of my life. I love them but I need to break free",
            "caught between my partner who wants to stay and my heart that wants to go. we have been together for 10 years",
        ],
    },
    "health": {
        "clear": [
            "I just got a scary diagnosis from my doctor",
            "I need to decide whether to have surgery",
            "My mental health is deteriorating",
            "I want to start therapy but I cannot afford it",
            "Should I get a second opinion on my diagnosis",
            "I am struggling with addiction",
            "My {relative} just got diagnosed with {condition}",
            "I have been ignoring symptoms for months",
            "I want to lose weight but nothing works",
            "My doctor wants me to start medication for {condition}",
            "I am having panic attacks every day",
            "Should I try alternative medicine",
            "I need to tell my family about my health condition",
            "My anxiety is getting worse and I cannot function",
            "I have been in chronic pain for years",
            "I want to quit smoking but I keep relapsing",
            "My depression is making it hard to work",
            "I am considering a major lifestyle change for my health",
            "Should I do the clinical trial",
            "I have not been to a doctor in years and I am scared",
            "My therapist says I need more intensive treatment",
            "I am exhausted all the time and do not know why",
            "I want to run a marathon but I have a heart condition",
            "My eating habits are out of control",
            "I need to take a medical leave from work",
        ],
        "subtle": [
            "I have not slept properly in months and it is affecting everything",
            "I keep putting off the appointment because I am scared of what they will find",
            "The pills help but I do not feel like myself anymore",
            "I drink every night now and I am not sure when that started",
            "My body is telling me something is wrong but I keep ignoring it",
            "I used to be able to run 5k and now I cannot climb stairs without getting winded",
            "I cannot concentrate on anything and my memory is getting worse",
            "Everyone says I look fine but I feel broken inside",
            "I had a miscarriage and I do not know how to process it",
            "My hands shake when I am sober and that was not happening a year ago",
        ],
        "messy": [
            "doctor says I need surgery within 3 months or it gets worse. but the recovery is 6 weeks and I cannot take that time off work. what do I even do",
            "I have been on antidepressants for 2 years and I want to stop but every time I try I crash. is this just going to be my life",
            "found a lump last week. have not told anyone. the appointment is tomorrow and I am terrified",
            "my dad was just diagnosed with early onset dementia. he is 62. I do not know how to hold it together for my family",
            "I know I drink too much. everyone knows. I have tried stopping three times this year. I am running out of excuses",
            "panic attack at work in front of everyone. now everyone treats me different. maybe I should just quit and deal with the anxiety first",
            "I have chronic fatigue and nobody believes me. not my doctor, not my family. I feel gaslit by my own body",
            "my kid has been diagnosed with ADHD and I am drowning in research about medication vs therapy vs diet. every source contradicts the other",
        ],
    },
    "finance": {
        "clear": [
            "Should I invest my savings in the stock market",
            "I am drowning in debt",
            "I want to buy a house but the market is insane",
            "Should I take out a loan to start a business",
            "I lost a lot of money on a bad investment",
            "My partner and I disagree about money",
            "I want to retire early but I am not sure I can afford it",
            "Should I put my money in crypto",
            "I need to declare bankruptcy",
            "My {relative} asked me to lend them money",
            "I have been living paycheck to paycheck",
            "Should I hire a financial advisor",
            "My business partner wants to take on more debt",
            "I want to save for my kids' education",
            "I inherited money and do not know what to do with it",
            "Should I sell my house in this market",
            "I am spending more than I earn",
            "My credit score is ruined",
            "I want to start investing but I know nothing about it",
            "Should I lease or buy a car",
            "I owe money to the tax authority",
            "My rent is going up and I cannot afford it",
            "I want to gamble my savings on a startup idea",
            "Should I cash out my retirement fund early",
            "I need to figure out how to budget properly",
        ],
        "subtle": [
            "The numbers do not add up at the end of the month and I do not know where the money goes",
            "I keep putting purchases on credit and telling myself I will pay it off next month",
            "My friends are all buying houses and I cannot even save for a deposit",
            "The market is crashing and I am watching my portfolio bleed",
            "I got a bonus and I cannot decide if I should invest it or enjoy it",
            "My parents never taught me about money and now I am 30 and clueless",
            "I am terrified of checking my bank account",
            "Inflation is eating my savings alive and I do not know how to outrun it",
            "I have three credit cards maxed out and a car payment I can barely make",
            "My co-founder wants to raise another round but I think we should be profitable first",
        ],
        "messy": [
            "so my landlord just raised the rent by 30%. I cannot afford it but moving is also expensive. I have maybe 2 months of savings left",
            "I put 50k into crypto two years ago and it is now worth 12k. do I sell and take the loss or keep hoping it recovers",
            "my wife wants us to buy a house but we have 80k in student loans. I say we pay the debt first. she says we are throwing away money on rent",
            "I am 28 and I have zero savings. I make decent money but it just disappears. I need a complete financial reset",
            "my business is barely breaking even after 3 years. everyone says give it one more year but I am burning through my retirement savings",
            "my father needs expensive care and he has no insurance. I love him but this is going to bankrupt me",
            "got offered a 40% raise to move to a city where everything costs twice as much. is it actually worth it",
            "I have been trading options and I am up 200% this year but I know it is basically gambling. when do I stop",
        ],
    },
    "education": {
        "clear": [
            "Should I drop out of college",
            "I want to go back to school at {age}",
            "I am failing my classes and might not graduate",
            "Should I pursue a PhD",
            "I cannot decide what to study",
            "My thesis advisor is terrible",
            "I want to study abroad for a semester",
            "Should I take a gap year",
            "I got rejected from my dream school",
            "I am considering switching my major",
            "My student loans are overwhelming",
            "I want to learn {skill} but I do not know where to start",
            "Should I get an MBA",
            "I am struggling to balance work and school",
            "My professor accused me of plagiarism",
            "I want to homeschool my kids",
            "Should I enroll in a bootcamp or get a degree",
            "I am not smart enough for this program",
            "My parents want me to study {subject} but I hate it",
            "I have to choose between two universities",
            "Should I do a postdoc or go into industry",
            "I want to become a researcher but the pay is bad",
            "I am too old to go back to school",
            "My grades are slipping and I do not care anymore",
            "Should I take online courses or attend in person",
        ],
        "subtle": [
            "I have a degree but I feel like I learned nothing useful",
            "Everyone in my cohort seems to get it except me",
            "I spent 4 years studying something I do not want to do",
            "The school system is failing my child and I do not know what the alternative is",
            "I keep signing up for courses and never finishing them",
            "My transcript is a mess from changing directions too many times",
            "I just realized my degree will not get me the job I actually want",
            "I aced the test but I cannot apply any of it in the real world",
            "I am halfway through a degree I chose for my parents not for myself",
            "I feel too stupid to be in this program but too stubborn to quit",
        ],
        "messy": [
            "I am 34 with two kids and I want to go back and get my degree but my wife says we cannot afford it. she is probably right but I feel stuck without it",
            "my PhD advisor is toxic. he takes credit for my work, blocks my publications, and makes me feel worthless. but if I leave now I lose 4 years",
            "I got into my dream program but they offered no funding. the other school gave me a full ride but it is ranked way lower. what matters more",
            "dropped out at 20 to start a business. the business failed. I am 27 now. is it too late to go back",
            "my kid is struggling in school and the teacher says he might have a learning disability. I do not know if I should get him tested or if she is just labeling him",
            "I have 120k in student loans for a degree I do not use. now I want to go back for something else. am I insane",
            "online degree vs in-person. the online one is half the price but will employers take it seriously. I genuinely do not know",
            "I am finishing my masters and I have no idea what to do next. academia is broken but industry feels like selling out",
        ],
    },
    "travel": {
        "clear": [
            "I want to move abroad but my family is here",
            "Should I relocate for a better life",
            "I am thinking about emigrating to {country}",
            "I got a job offer in another country",
            "Should I travel solo around the world",
            "I want to live in {city} but I do not speak the language",
            "My partner wants to move but I want to stay",
            "I am homesick and want to go back",
            "Should I move to a cheaper city",
            "I want to leave everything behind and start fresh",
            "My visa is expiring and I need to decide what to do",
            "I am afraid of flying but I need to travel",
            "Should I move closer to my family",
            "I want to live as a digital nomad",
            "My company wants me to transfer overseas",
            "I moved to a new city and I have no friends",
            "Should I go back to my home country",
            "I want to experience a different culture",
            "I am stuck between two cities",
            "My kids do not want to move",
            "I want to retire abroad",
            "Should I move to the countryside or stay in the city",
            "I want to immigrate but the process is overwhelming",
            "I have been traveling for a year and I feel lost",
            "Should I settle down or keep moving",
        ],
        "subtle": [
            "I do not feel like I belong here anymore",
            "This city has nothing left for me but leaving means starting over",
            "I keep googling apartments in {city} at 3am",
            "My lease is up in two months and I keep thinking about just not renewing it",
            "Home does not feel like home since I came back",
            "I have been living out of a suitcase for six months and I love it but I am also exhausted",
            "All my friends from university scattered to different countries and I am the only one who stayed",
            "I need a change of scenery but I am not sure if I mean a vacation or a move",
            "The winters here are destroying me. I need sun or I am going to lose it",
            "I have dual citizenship and I keep going back and forth about which country to settle in",
        ],
        "messy": [
            "I want to live abroad but all my friends and family are in Hong Kong. I love it here but I feel like there is a bigger world out there",
            "got an offer to work in Singapore. double my salary. but my parents are aging and I am the only child. I feel selfish for even considering it",
            "I moved to {city} for love and the relationship ended. now I am stuck in a country where I have nothing except a lease I cannot break",
            "my wife and I always said we would live abroad for a few years. now we have a 2-year-old and she says it is too risky. I feel like the window is closing",
            "I have been a digital nomad for 3 years and I am burned out. everyone thinks my life is amazing but I am lonely and exhausted. I want roots but I do not know where",
            "I left my country because of the political situation and I miss it every single day. I do not know if it is safe to go back",
            "spent my whole life in the same small town. I am 40. is it too late to move to a city and actually experience life",
            "I keep moving every 2 years because I get bored. maybe the problem is not the place but me. but also maybe I just have not found the right place yet",
        ],
    },
}

# emotion signal patterns: both prefixes and suffixes that carry clear
# emotional signal.  Using both in combination trains the model to detect
# emotion from context rather than just the first few words.
EMOTION_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "anxious": {
        "prefix": [
            "I am worried about", "I am nervous that", "I am scared to",
            "I am afraid of", "I am terrified that", "I am stressed about",
            "I cannot stop worrying about", "I am panicking about",
            "I am hesitant to", "I am unsure whether to",
        ],
        "suffix": [
            "and I cannot sleep thinking about it",
            "and my anxiety is through the roof",
            "and I feel sick to my stomach just thinking about it",
            "and I keep imagining the worst case",
            "and I am paralyzed by the uncertainty",
        ],
    },
    "confident": {
        "prefix": [
            "I am ready to", "I am determined to", "I know I should",
            "I am certain that", "I am sure I want to", "I feel strong about",
            "I have decided to", "I am bold enough to",
            "I finally have the courage to", "I am done waiting to",
        ],
        "suffix": [
            "and I know I can handle whatever comes",
            "and I am more sure of this than anything",
            "and honestly I am excited about it",
            "and nothing is going to stop me",
            "and I feel like it is now or never in a good way",
        ],
    },
    "angry": {
        "prefix": [
            "I am furious that", "I am angry about", "I am frustrated with",
            "I am upset that", "I cannot believe", "I am fed up with",
            "I am resentful about", "I am mad that",
            "I am livid about", "I am outraged by",
        ],
        "suffix": [
            "and I am done being patient about it",
            "and I want to burn it all down",
            "and someone needs to be held accountable",
            "and I feel disrespected and ignored",
            "and I swear if this happens one more time I am going to explode",
        ],
    },
    "hopeful": {
        "prefix": [
            "I am excited about", "I am optimistic that", "I look forward to",
            "I am eager to", "I am enthusiastic about", "I hope that",
            "I think things are about to get better because",
            "I have a good feeling about",
        ],
        "suffix": [
            "and I think this could change everything for the better",
            "and for the first time in a long time I feel like things will work out",
            "and I am cautiously optimistic about where this is heading",
            "and maybe this is the fresh start I needed",
            "and I feel a sense of possibility I have not felt in years",
        ],
    },
    "desperate": {
        "prefix": [
            "I am desperate to", "I feel hopeless about", "I am stuck with",
            "I feel trapped by", "I have no choice but to", "this is my last resort with",
            "I have tried everything and", "I do not know what else to do about",
        ],
        "suffix": [
            "and I feel like I am running out of options",
            "and I do not see a way out",
            "and honestly I feel like I am drowning",
            "and I am at the end of my rope",
            "and I do not know how much longer I can take this",
        ],
    },
    "neutral": {
        "prefix": [
            "I am thinking about", "I am considering", "should I",
            "I need to decide whether to", "I want to",
            "I am weighing the pros and cons of",
            "I have been going back and forth about",
            "I need advice about",
        ],
        "suffix": [
            "",
            "and I want to think it through carefully",
            "and I am not sure which way to lean",
        ],
    },
}

FILL_VALUES: dict[str, list[str]] = {
    "field": ["tech", "medicine", "law", "education", "consulting", "art",
              "data science", "marketing", "engineering", "nonprofits"],
    "company": ["Google", "a startup", "a rival firm", "a nonprofit",
                "a hospital", "a hedge fund", "a mid-size company", "a remote company"],
    "profession": ["developer", "nurse", "chef", "writer", "pilot", "designer",
                   "therapist", "teacher", "architect", "photographer"],
    "relative": ["mother", "father", "brother", "sister", "cousin", "uncle",
                 "grandmother", "partner", "husband", "wife"],
    "condition": ["cancer", "diabetes", "ADHD", "chronic fatigue", "MS",
                  "bipolar disorder", "celiac disease", "endometriosis"],
    "age": ["28", "32", "35", "40", "45", "50", "55"],
    "country": ["Japan", "Canada", "Australia", "Germany", "Singapore", "Portugal",
                "New Zealand", "Netherlands", "South Korea", "Taiwan", "UK"],
    "city": ["Tokyo", "London", "New York", "Berlin", "Bangkok", "Lisbon",
             "Melbourne", "Vancouver", "Amsterdam", "Taipei", "Barcelona"],
    "subject": ["engineering", "medicine", "law", "business", "science",
                "accounting", "computer science", "psychology"],
    "skill": ["programming", "data science", "photography", "music", "cooking",
              "UX design", "machine learning", "creative writing"],
}

MODIFIERS: list[str] = [
    "but I am not sure it is the right time",
    "and everyone is telling me not to",
    "because things cannot stay the way they are",
    "even though it scares me",
    "and I need to decide soon",
    "but my family disagrees",
    "and the stakes are really high",
    "but I keep going back and forth",
    "and I feel like time is running out",
    "but part of me thinks I should just stay the course",
    "and I have been overthinking this for months",
    "",
    "",
    "",
    "",
]

# cross-domain hard examples: scenarios that sound like one domain but are
# actually another.  These force the model to look past surface keywords.
CROSS_DOMAIN_HARD: list[dict] = [
    {"text": "I want to move to {city} for a job opportunity", "domain": "travel", "emotion": "hopeful"},
    {"text": "my relationship is suffering because of my work schedule", "domain": "relationship", "emotion": "desperate"},
    {"text": "I cannot afford the medical treatment I need", "domain": "health", "emotion": "desperate"},
    {"text": "my student loans are preventing me from buying a house", "domain": "finance", "emotion": "anxious"},
    {"text": "I dropped out to take care of my sick parent", "domain": "education", "emotion": "desperate"},
    {"text": "I want to study abroad but I would miss my partner", "domain": "education", "emotion": "anxious"},
    {"text": "the stress from my job is giving me health problems", "domain": "health", "emotion": "anxious"},
    {"text": "I moved to a new country and my marriage is falling apart", "domain": "relationship", "emotion": "desperate"},
    {"text": "I spent all my savings on a degree that is worthless", "domain": "finance", "emotion": "angry"},
    {"text": "I want to quit my job and travel the world", "domain": "travel", "emotion": "hopeful"},
    {"text": "my health insurance is too expensive and I might have to drop it", "domain": "finance", "emotion": "anxious"},
    {"text": "I got a scholarship abroad but my family needs me here", "domain": "education", "emotion": "anxious"},
    {"text": "I want to live abroad but all my friends and family are in Hong Kong", "domain": "travel", "emotion": "anxious"},
    {"text": "my boss is making me so stressed I am getting chest pains", "domain": "health", "emotion": "desperate"},
    {"text": "I want to start a business with my partner but I am afraid it will ruin us", "domain": "career", "emotion": "anxious"},
    {"text": "the cost of living in {city} is destroying my quality of life", "domain": "finance", "emotion": "desperate"},
    {"text": "I keep changing jobs and my resume looks terrible", "domain": "career", "emotion": "anxious"},
    {"text": "I need to decide between paying for my kid's school or my retirement", "domain": "finance", "emotion": "anxious"},
    {"text": "my partner got a job offer overseas and expects me to follow", "domain": "travel", "emotion": "angry"},
    {"text": "I graduated but I cannot find a job in my field", "domain": "career", "emotion": "desperate"},
    {"text": "the doctor said I need to quit my high-stress job for my heart", "domain": "health", "emotion": "anxious"},
    {"text": "I want to retrain in a completely new field at {age}", "domain": "education", "emotion": "hopeful"},
    {"text": "my friend owes me money and it is ruining our friendship", "domain": "relationship", "emotion": "angry"},
    {"text": "I keep spending money I do not have on things I do not need", "domain": "finance", "emotion": "desperate"},
    {"text": "I am homesick but there are no jobs back home", "domain": "travel", "emotion": "desperate"},
    {"text": "my parents sacrificed everything for my education and I am failing", "domain": "education", "emotion": "desperate"},
    {"text": "I want to move closer to my aging parents but my career is here", "domain": "travel", "emotion": "anxious"},
    {"text": "I am in love with someone from another country and one of us has to move", "domain": "relationship", "emotion": "anxious"},
    {"text": "my whole identity is my job and I just got fired", "domain": "career", "emotion": "desperate"},
    {"text": "I need a break from everything but I cannot afford to stop working", "domain": "health", "emotion": "desperate"},
]


def _fill_template(template: str) -> str:
    for match in re.finditer(r"\{(\w+)\}", template):
        key = match.group(1)
        if key in FILL_VALUES:
            template = template.replace(match.group(0), random.choice(FILL_VALUES[key]), 1)
    return template


def _apply_emotion_to_text(text: str, emotion: str, position: str) -> str:
    templates = EMOTION_TEMPLATES.get(emotion, EMOTION_TEMPLATES["neutral"])

    if position == "prefix":
        prefix = random.choice(templates["prefix"])
        return f"{prefix} {text[0].lower()}{text[1:]}"
    elif position == "suffix":
        suffix = random.choice(templates["suffix"])
        if suffix:
            return f"{text} {suffix}"
        return text
    elif position == "both":
        prefix = random.choice(templates["prefix"])
        suffix = random.choice(templates["suffix"])
        result = f"{prefix} {text[0].lower()}{text[1:]}"
        if suffix:
            result = f"{result} {suffix}"
        return result
    return text


def generate_dataset(n_samples: int = 5000,
                     output_path: str | None = None,
                     seed: int = 42) -> list[dict]:
    random.seed(seed)
    samples: list[dict] = []

    domains = list(DOMAIN_TEMPLATES.keys())
    emotions = list(EMOTION_TEMPLATES.keys())

    # 70% from tiered domain templates, 15% cross-domain hard, 15% emotion-augmented
    n_domain_samples = int(n_samples * 0.70)
    n_cross_domain = int(n_samples * 0.15)
    n_emotion_augmented = n_samples - n_domain_samples - n_cross_domain

    per_domain = n_domain_samples // len(domains)
    remainder = n_domain_samples % len(domains)

    for domain in domains:
        count = per_domain + (1 if remainder > 0 else 0)
        remainder -= 1

        tier_weights = {"clear": 0.45, "subtle": 0.30, "messy": 0.25}
        tiers = list(DOMAIN_TEMPLATES[domain].keys())

        for _ in range(count):
            tier = random.choices(tiers, weights=[tier_weights[t] for t in tiers])[0]
            templates = DOMAIN_TEMPLATES[domain][tier]
            base = _fill_template(random.choice(templates))
            emotion = random.choice(emotions)

            r = random.random()
            if emotion != "neutral" and r < 0.3:
                text = _apply_emotion_to_text(base, emotion, "prefix")
            elif emotion != "neutral" and r < 0.5:
                text = _apply_emotion_to_text(base, emotion, "suffix")
            elif emotion != "neutral" and r < 0.6:
                text = _apply_emotion_to_text(base, emotion, "both")
            else:
                text = base

            if random.random() < 0.35:
                modifier = random.choice(MODIFIERS)
                if modifier:
                    text = f"{text} {modifier}"

            samples.append({"text": text, "domain": domain, "emotion": emotion})

    for _ in range(n_cross_domain):
        template = random.choice(CROSS_DOMAIN_HARD)
        text = _fill_template(template["text"])
        emotion = template["emotion"]

        if random.random() < 0.4 and emotion != "neutral":
            pos = random.choice(["prefix", "suffix"])
            text = _apply_emotion_to_text(text, emotion, pos)

        if random.random() < 0.25:
            modifier = random.choice(MODIFIERS)
            if modifier:
                text = f"{text} {modifier}"

        samples.append({
            "text": text,
            "domain": template["domain"],
            "emotion": emotion,
        })

    for _ in range(n_emotion_augmented):
        domain = random.choice(domains)
        emotion = random.choice([e for e in emotions if e != "neutral"])
        tier = random.choice(list(DOMAIN_TEMPLATES[domain].keys()))
        base = _fill_template(random.choice(DOMAIN_TEMPLATES[domain][tier]))
        text = _apply_emotion_to_text(base, emotion, "both")

        samples.append({"text": text, "domain": domain, "emotion": emotion})

    random.shuffle(samples)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(samples, f, indent=2)

    return samples


if __name__ == "__main__":
    data = generate_dataset(
        n_samples=5000,
        output_path="data/training_data.json",
    )
    print(f"Generated {len(data)} samples")

    domain_counts: dict[str, int] = {}
    emotion_counts: dict[str, int] = {}
    for s in data:
        domain_counts[s["domain"]] = domain_counts.get(s["domain"], 0) + 1
        emotion_counts[s["emotion"]] = emotion_counts.get(s["emotion"], 0) + 1

    print("\nDomain distribution:")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d:15s} {c:5d} ({c / len(data) * 100:.1f}%)")

    print("\nEmotion distribution:")
    for e, c in sorted(emotion_counts.items()):
        print(f"  {e:15s} {c:5d} ({c / len(data) * 100:.1f}%)")

    print("\nSample examples:")
    for sample in data[:8]:
        print(f"  [{sample['domain']:12s}] [{sample['emotion']:10s}] {sample['text'][:100]}")
