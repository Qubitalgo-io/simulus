from __future__ import annotations

import json
import random
from pathlib import Path


DOMAIN_TEMPLATES: dict[str, list[str]] = {
    "career": [
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
    "relationship": [
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
    "health": [
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
    "finance": [
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
    "education": [
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
    "travel": [
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
}

EMOTION_TEMPLATES: dict[str, list[str]] = {
    "anxious": [
        "I am worried about", "I am nervous that", "I am scared to",
        "I am afraid of", "I am terrified that", "I am stressed about",
        "I cannot stop worrying about", "I am panicking about",
        "I am hesitant to", "I am unsure whether to",
    ],
    "confident": [
        "I am ready to", "I am determined to", "I know I should",
        "I am certain that", "I am sure I want to", "I feel strong about",
        "I have decided to", "I am bold enough to",
    ],
    "angry": [
        "I am furious that", "I am angry about", "I am frustrated with",
        "I am upset that", "I cannot believe", "I am fed up with",
        "I am resentful about", "I am mad that",
    ],
    "hopeful": [
        "I am excited about", "I am optimistic that", "I look forward to",
        "I am eager to", "I am enthusiastic about", "I hope that",
    ],
    "desperate": [
        "I am desperate to", "I feel hopeless about", "I am stuck with",
        "I feel trapped by", "I have no choice but to", "this is my last resort",
    ],
    "neutral": [
        "I am thinking about", "I am considering", "should I",
        "I need to decide whether to", "I want to",
    ],
}

FILL_VALUES: dict[str, list[str]] = {
    "field": ["tech", "medicine", "law", "education", "consulting", "art"],
    "company": ["Google", "a startup", "a rival firm", "a nonprofit"],
    "profession": ["developer", "nurse", "chef", "writer", "pilot", "designer"],
    "relative": ["mother", "father", "brother", "sister", "cousin", "uncle"],
    "condition": ["cancer", "diabetes", "ADHD", "chronic fatigue"],
    "age": ["30", "35", "40", "45", "50"],
    "country": ["Japan", "Canada", "Australia", "Germany", "Singapore", "Portugal"],
    "city": ["Tokyo", "London", "New York", "Berlin", "Bangkok", "Lisbon"],
    "subject": ["engineering", "medicine", "law", "business", "science"],
    "skill": ["programming", "data science", "photography", "music", "cooking"],
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
    "",
    "",
    "",
]


def _fill_template(template: str) -> str:
    import re
    for match in re.finditer(r"\{(\w+)\}", template):
        key = match.group(1)
        if key in FILL_VALUES:
            template = template.replace(match.group(0), random.choice(FILL_VALUES[key]), 1)
    return template


def generate_dataset(n_samples: int = 2000,
                     output_path: str | None = None,
                     seed: int = 42) -> list[dict]:
    random.seed(seed)
    samples = []

    domains = list(DOMAIN_TEMPLATES.keys())
    emotions = list(EMOTION_TEMPLATES.keys())

    per_domain = n_samples // len(domains)
    remainder = n_samples % len(domains)

    for domain in domains:
        count = per_domain + (1 if remainder > 0 else 0)
        remainder -= 1

        templates = DOMAIN_TEMPLATES[domain]

        for _ in range(count):
            base = _fill_template(random.choice(templates))
            emotion = random.choice(emotions)

            # sometimes prepend an emotion prefix
            if random.random() < 0.4 and emotion != "neutral":
                prefix = random.choice(EMOTION_TEMPLATES[emotion])
                text = f"{prefix} {base[0].lower()}{base[1:]}"
            else:
                text = base

            # sometimes append a modifier
            modifier = random.choice(MODIFIERS)
            if modifier:
                text = f"{text} {modifier}"

            samples.append({
                "text": text,
                "domain": domain,
                "emotion": emotion,
            })

    random.shuffle(samples)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(samples, f, indent=2)

    return samples


if __name__ == "__main__":
    data = generate_dataset(
        n_samples=2000,
        output_path="data/training_data.json",
    )
    print(f"Generated {len(data)} samples")
    for sample in data[:5]:
        print(f"  [{sample['domain']:12s}] [{sample['emotion']:10s}] {sample['text']}")
