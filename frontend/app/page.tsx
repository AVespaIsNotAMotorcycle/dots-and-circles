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

const BACKEND = 'http://localhost:5000';

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
				setLexigraphClass(data.class);
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
			<LoadButton setWord={setWord} setFont={setFont} />
			<div style={{ display: 'flex', justifyContent: 'space-between' }}>
				<div>
					<LexigraphClassDescription lexigraphClass={lexigraphClass} />
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
				Dots and Circles is my (Sasha Madden Ebersole's) first non-tutorial machine learning
				project. It is an app for optical character recognition of Manchu-language text. An
				image is converted to a string of Unicode text in four stages:
			</p>
			<dl>
				<div>
  				<dt>Classifier</dt>
  				<dd>
  					This is a Recurrant Neural Network which reads an image line by line and then sorts
  					it into one of four categories which correspond to various styles of Manchu writing.
  				</dd>
				</div>
				<div>
				<dt>Primary OCR</dt>
				<dd>
					This is a Neural Network which, given 21 sequential rows of pixels, identifies the
					Manchu letter represented by the middle row. There are four different objects of this
					class, each corresponding to and trained on one of the four categories identified by
					the Classifier.
				</dd>
				</div>
				<div>
  				<dt>Secondary OCR</dt>
  				<dd>
  					This is a Neural Network which, given 21 sequential outputs from the primary OCR,
  					identifies the Manchu letter represented by the middle row. This was added to handle
  					misidentifications by the Primary OCR; though the Primary OCR often got letters wrong,
  					it tended to get them wrong in particular patterns, which the Secondary OCR can
  					recognize and account for. Much like the Primary OCR, this has four specialized variants.
  				</dd>
				</div>
				<div>
  				<dt>CTC Parser</dt>
  				<dd>
  					This takes the output from the Secondary OCR, which is much longer than the actual word,
  					and converts it into a string. While doing so, it also accounts for Manchu phonotactics
  					to remove invalid letters. Currently, no machine learning is used here.
  				</dd>
				</div>
			</dl>
			<p>
				Training data is, as of now, entirely synthetic, generated by rendering Manchu unicode text
				from <a href={manchuCakeLink}>manchu-cake</a> in various fonts.
				At some point I intend to revisit this project
				with a new approach, at which point I will also use real-world data. A set of real-world data
				was compiled and published by Sun Haipeng, Tao Wenhao, and Bi Xiaojun and is
				available <a href={datasetLink}>here</a>, which I will likely use for the next iteration on this
				project.
			</p>
		</section>
	);
}

export default function Home() {
	return (
		<main className={styles.homepage}>
			<h1>Dots and Circles</h1>
			<About />
			<Demo />
		</main>
	);
}
