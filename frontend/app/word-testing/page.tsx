'use client'

import axios from 'axios';
import { useState, useEffect } from 'react';
import levenshtein from 'js-levenshtein';

import styles from './word-testing.module.css';
import OCRVisualization from '../ocr-visualization';

const BACKEND = 'http://localhost:5000';
const fonts = [
		'BiaoHei',
		'GuFeng',
		'LiuYe',
		'ShuKai',
		'WenJian',
		'WenQin',
		'XingShu',
		'YaBai',
		'YingBi',
		'ZhengBai',
		'ZhengHei',
	];

function LexigraphClassDescription({ lexigraphClass }) {
	const traits = ['Lexigraphs of this class have heavier line weight.',
									'Lexigraphs of this class have lighter line weight.',
									'The letters <M> and <L> are connected to the center line.',
									'The letters <M> and <L> are disconnected from the center line.',
									'The letters <A> and <E> are are pointy.',
									'The letters <A> and <E> are are rounded.'];
	let description = '';

	switch (lexigraphClass) {
		case 'A':
			description = [0, 2, 4].map((index) => traits[index]).join(' ');
			break;
		case 'B':
			description = [0, 2, 5].map((index) => traits[index]).join(' ');
			break;
		case 'C':
			description = [1, 3, 5].map((index) => traits[index]).join(' ');
			break;
		case 'D':
			description = [0, 3, 5].map((index) => traits[index]).join(' ');
			break;
		default:
			description = '';
	}

	return (
		<section className={[styles.classDescription, styles[`class${lexigraphClass}`]].join (' ')}>
			<h2>{`Class ${lexigraphClass}`}</h2>
			<p>{description}</p>
		</section>
	);
}

function Performance({ word, prediction }) {
	const degree = levenshtein(word, prediction) / word.length;

	let degreeClass = 'bad';
	const cutoffs = [0.25, 1];
	if (degree < cutoffs[0]) degreeClass = 'good';
	if (degree < cutoffs[1]) degreeClass = 'mid';

	const description = ['The degree of error is the Levenshtein distance between the actual string',
											 'and the string predicted by the neural network, divided by the length',
											 'of the actual string.',
											 `A score below ${cutoffs[0]} is good.`,
											 `A score above that but below ${cutoffs[1]} is okay.`,
											 `A score above ${cutoffs[1]} is bad.`].join(' ');

	return (
		<div className={[styles.performance, styles[degreeClass]].join(' ')}>
			<div className={styles.degree}>
				{`Degree of error: ${degree.toPrecision(2)}`}
			</div>
			<p>
				{description}
			</p>
		</div>
	)
}

function LoadButton({ setWord, setFont }) {
	const [fonts, setFonts] = useState({});
	
	const onClick = () => {
		axios.get(`${BACKEND}/corpus/random`)
			.then(({ data }) => {
				setWord(data.manchu);
				const fontIndex = Math.floor(Math.random() * Object.keys(fonts).length);
				const fontKey = Object.keys(fonts)[fontIndex];
				console.log(fontIndex, fontKey);
				setFont(fontKey);
			});
	};

	useEffect(() => {
		axios.get(`${BACKEND}/lexigraphy/fonts/dict`)
			.then(({ data }) => { setFonts(data); onClick(); });
	}, []);

	return <button type="button" onClick={onClick}>LOAD RANDOM WORD</button>;
}

export default function WordTesting({}) {
	const [word, setWord] = useState('');
	const [font, setFont] = useState(0);

	const [lexigraphClass, setLexigraphClass] = useState();
	const [predictions, setPredictions] = useState([]);

	const [parsed, setParsed] = useState('');

	const url = `${BACKEND}/lexigraphy/new/${font}/${word}`;

	useEffect(() => {
		if (font === undefined) return;
		if (!word) return;
		axios.get(`${BACKEND}/lexigraphy/predict/${font}/${word}`)
			.then(({ data }) => {
				console.log(data.class);
				setLexigraphClass(data.class);
				setPredictions(data.predictions);
			})
			.catch(console.error);
	}, [word, font]);

	return (
		<main>
			<h1>Word Testing</h1>
			<div>
			</div>
			<div className={styles.classSection}>
				<img src={url} />
				<div>
					<LoadButton setWord={setWord} setFont={setFont} />
					<LexigraphClassDescription lexigraphClass={lexigraphClass} />
				</div>
				<OCRVisualization font={font} word={word} results={predictions} setParsed={setParsed} />
				<Performance word={word} prediction={parsed} />
			</div>
			<table>
				<thead>
					<tr>
						<th>Feature</th>
						{fonts.map((name) => <th>{name}</th>)}
					</tr>
				</thead>
				<tbody>
					<tr>
						<td />
						{fonts.map((name, index) => (
							<td
								className="manchu-text"
								key={name}
								style={{ fontFamily: `manchu${index}`, fontSize: '3rem' }}
							>
								{word}
							</td>
						))}
					</tr>
					<tr>
						<td>Thick lines</td>
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'pink' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'pink' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
					</tr>
					<tr>
						<td>m/l connected to main body</td>
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'pink' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'pink' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'pink' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
					</tr>
					<tr>
						<td>e/a are pointy</td>
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'pink' }} />
						<td style={{ background: 'pink' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'pink' }} />
						<td style={{ background: 'pink' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'pink' }} />
						<td style={{ background: 'lightgreen' }} />
						<td style={{ background: 'lightgreen' }} />
					</tr>
					<tr>
						<td>Class</td>
						<td>A</td>
						<td>A</td>
						<td>B</td>
						<td>C</td>
						<td>A</td>
						<td>B</td>
						<td>D</td>
						<td>A</td>
						<td>C</td>
						<td>A</td>
						<td>A</td>
					</tr>
				</tbody>
			</table>
		</main>
	)
}
