'use client'

import axios from 'axios';
import { useState, useEffect } from 'react';

import styles from './main.module.css';

import LexigraphClassDescription from './components/LexigraphClassDescription';
import Performance from './components/Performance';
import OCRVisualization, {
	removeInvalidDipthongs,
	enforceVowelHarmony,
	parseResults,
	RowPredictionVisualizer,
	PredictionChartLegend,
} from './components/OCRVisualization';

const BACKEND = process.env.NEXT_PUBLIC_BACKEND

function LoadButton({ setWord, setFont }) {
	const [fonts, setFonts] = useState({});
	
	const onClick = () => {
		axios.get(`${BACKEND}/corpus/random`)
			.then(({ data }) => {
				setWord(data.manchu);
				const fontIndex = Math.floor(Math.random() * Object.keys(fonts).length);
				const fontKey = Object.keys(fonts)[fontIndex];
				setFont(fontKey);
			});
	};

	useEffect(() => {
		axios.get(`${BACKEND}/lexigraphy/fonts/dict`)
			.then(({ data }) => { setFonts(data); });
	}, []);

	return <button type="button" style={{ marginBottom: '1rem' }} onClick={onClick}>LOAD RANDOM WORD</button>;
}

function Demo() {
	const [word, setWord] = useState();
	const [font, setFont] = useState();
	const [lexigraphClass, setLexigraphClass] = useState('A');

	const [primaryPredictions, setPrimaryPredictions] = useState([]);
	const [secondaryPredictions, setSecondaryPredictions] = useState([]);
	const [prediction, setPrediction] = useState('');

	const url = `${BACKEND}/lexigraphy/new/${font}/${word}`;

	useEffect(() => {
		if (font === undefined) return;
		if (!word) return;
		axios.get(`${BACKEND}/lexigraphy/predict/${font}/${word}`)
			.then(({ data }) => {
				setLexigraphClass(data.l_class);
				setPrimaryPredictions(data.primary_predictions);
				setSecondaryPredictions(data.secondary_predictions);

				const parsed = parseResults(data.secondary_predictions);
				const harmonious = enforceVowelHarmony(parsed);
				const validDipthongs = removeInvalidDipthongs(harmonious);
				setPrediction(validDipthongs);
			})
			.catch(console.error);
	}, [word, font]);

	if (!word) {
		return (
			<section>
				<h2>Demo</h2>
				<LoadButton setWord={setWord} setFont={setFont} />
			</section>
		);
	}
	return (
		<section>
			<h2>Demo</h2>
			<div className={styles.demoWrapper}>
				<div>
					<LoadButton setWord={setWord} setFont={setFont} />
					{lexigraphClass && <LexigraphClassDescription lexigraphClass={lexigraphClass} />}
				</div>
				<div className={styles.visualizerWrapper}>
					<h3>Primary OCR</h3>
					<h3>Input Image</h3>
					<h3>Secondary OCR</h3>
					<h3>Parsed Output</h3>
					<RowPredictionVisualizer prediction={primaryPredictions} flip />
					<img className={styles.lexigraph} src={url} />
					<RowPredictionVisualizer prediction={secondaryPredictions} />
					<p
						className="manchu-text"
						style={{ marginTop: '20px', fontSize: '3.8rem', fontFamily: `manchu${font}` }}
					>
						{prediction}
	  			</p>
				</div>
				<Performance word={word} prediction={prediction} />
			</div>
		</section>
	);
}

function About() {
	const manchuCakeLink = 'https://github.com/OverflowCat/manchu-cake';
	const datasetLink = 'https://www.scidb.cn/en/detail?dataSetId=b45491b63d694534a9323acf14846586';
	return (
		<section>
			<h2>About</h2>
			<p>
				Dots and Circles is my first non-tutorial machine learning project.
				It is an app for optical character recognition of Manchu-language text.
			</p>
			<section>
				<h3>The Problem</h3>
				<p>
					The Manchu script is written top-to-bottom, left-to-right. All letters within a word
					are connected to each other, and the shape of a letter changes based on its position
					within a word. This makes it quite difficult to break a word down into individual
					letters. In order to circumvent this, I decided to procede using connectionist
					temporal classification (CTC), or at least my vague impression of CTC.
				</p>
			</section>
			<section>
				<h3>The Approach</h3>
				<p>
					Training data was generated synthetically using
					the <a href={"https://github.com/OverflowCat/manchu-cake"}>manchu-cake</a> dictionary.
					Each unique manchu word in the dictionary was stored in a SQLite database, and random
					words were pulled from the database and rendered in a random font. Each image of a word
					in a particular font (called a <span className="bold">lexigraph</span> from here on)
					had letter boundaries manually labelled. Each lexigraph is 50 pixels wide, and there
					are 10 pixels of whitespace on each side.
				</p>
				<p>
					OCR is performed by two neural networks. The first networks, henceforth called
					the <span className="bold">Primary OCR</span>, takes as input 21 sequential rows of pixels
					in a lexigraph and outputs a single character. This character is what the Primary OCR
					thinks the middle row of pixels corresponds to.
				</p>
				<p>
					Unfortunately, the Primary OCR had difficulty differentiating certain letters. However,
					it tended to produce errors in patterns. In order to counteract this, I added a second
					network, called the <span className="bold">Secondary OCR</span>, which took as input 21
					outputs from the Primary OCR and output the character it predicted the middle to actually
					correspond to. This had the effect of reducing noise in output and marginally improving
					accuracy.
				</p>
			</section>
			<section>
				<h3>Lessons Learned</h3>
				<p>
					At some point in the future I aim to redo this project. When I do, I will likely use
					an encoder/decoder model, with an RNN (the encoder) first reading the lexigraph line
					by line and producing some representation of the word, and a second RNN (the decoder)
					using that representation to spell out the word character by character.
				</p>
				<p>
					Additionally, I abandon the marking of character boundaries in favor of giving the
					machine whole lexigraphs and allowing it to learn the boundaries on its own. This would
					make it easier to generate training data, and make it easier to incorporate real-world
					data. There is, for example, a publically available{' '}
					<a href={"https://www.scidb.cn/en/detail?dataSetId=b45491b63d694534a9323acf14846586"}>
						dataset
					</a>
					{' '}of words scanned from Manchu books published by Sun Haipeng, Tao Wenhao, and Bi
					Xiaojun. In this future version, I will use a test set composed entirely of real-world
					data, while the training set would be a mix of synthetic and real-world lexigraphs.
				</p>
			</section>
		</section>
	);
}

export default function Home() {
	return (
		<main className={styles.homepage}>
			<h1>Dots and Circles</h1>
			<Demo />
			<About />
		</main>
	);
}
