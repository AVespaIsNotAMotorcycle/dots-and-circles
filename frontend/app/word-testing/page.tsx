'use client'

import axios from 'axios';
import { useState, useEffect } from 'react';
import levenshtein from 'js-levenshtein';

import styles from './word-testing.module.css';
import OCRVisualization, {
	removeInvalidDipthongs,
	enforceVowelHarmony,
	parseResults,
} from '../components/OCRVisualization';

const BACKEND = process.env.NEXT_PUBLIC_BACKEND

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

function accuracy(word, prediction) {
	const sumOfLengths = word.length + prediction.length;
	const distance = levenshtein(word, prediction);
	const degree = (sumOfLengths - distance) / sumOfLengths;
	return degree;
}

function Performance({ word, prediction }) {
	const degree = accuracy(word, prediction);

	let degreeClass = 'bad';
	const cutoffs = [0.7, 0.9];
	if (degree > cutoffs[0]) degreeClass = 'mid';
	if (degree > cutoffs[1]) degreeClass = 'good';

	const description = ['Accuracy is (s - d) / s, wehere s is the sum of the lengths',
											 'of the actual string and the string predicted by the network',
											 'and d is the Levenshtein distance between the two.',
											 `A score below ${cutoffs[0]} is bad.`,
											 `A score above that but below ${cutoffs[1]} is okay.`,
											 `A score above ${cutoffs[1]} is good.`,
											 'A score of 1 is perfect.'].join(' ');

	return (
		<div className={[styles.performance, styles[degreeClass]].join(' ')}>
			<div className={styles.degree}>
				{`Accuracy: ${degree.toPrecision(2)}`}
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
				setFont(fontKey);
			});
	};

	useEffect(() => {
		axios.get(`${BACKEND}/lexigraphy/fonts/dict`)
			.then(({ data }) => { setFonts(data); onClick(); });
	}, []);

	return <button type="button" onClick={onClick}>LOAD RANDOM WORD</button>;
}

function AverageDegreesOfError({ records }) {
	const sums = records.reduce((accumulator, currentValue) => {
		accumulator[0] = accumulator[0] + accuracy(currentValue.word, currentValue.parsed);
		accumulator[1] = accumulator[1] + accuracy(currentValue.word, currentValue.harmonious);
		accumulator[2] = accumulator[2] + accuracy(currentValue.word, currentValue.validDipthongs);
		return accumulator;
	}, [0, 0, 0]);
	if (!sums) return null;
	return (
		<table>
			<thead>
				<tr>
					<th>parse()</th>
					<th>enforceVowelHarmony()</th>
					<th>removeInvalidDipthongs()</th>
				</tr>
			</thead>
			<tbody>
				<tr>
					<td>{(sums[0] / records.length).toPrecision(2)}</td>
					<td>{(sums[1] / records.length).toPrecision(2)}</td>
					<td>{(sums[2] / records.length).toPrecision(2)}</td>
				</tr>
			</tbody>
		</table>
	)
}

function Autocorrect({ word }) {
	const [correction, setCorrection] = useState(word);

	useEffect(() => {
		axios.get(`${BACKEND}/corpus/autocorrect/${word}`)
			.then(({ data }) => { setCorrection(data.word); });
	}, [word]);

	return <p className="manchu-text">{correction}</p>;
}

export default function WordTesting({}) {
	const [word, setWord] = useState('');
	const [font, setFont] = useState(0);

	const [lexigraphClass, setLexigraphClass] = useState();
	const [predictions, setPredictions] = useState([]);

	const [parsed, setParsed] = useState('');

	const [records, setRecords] = useState([]);

	const url = `${BACKEND}/lexigraphy/new/${font}/${word}`;

	useEffect(() => {
		if (font === undefined) return;
		if (!word) return;
		axios.get(`${BACKEND}/lexigraphy/predict/${font}/${word}`)
			.then(({ data }) => {
				setLexigraphClass(data.class);
				setPredictions(data.predictions);

				const parsed = parseResults(data.predictions);
				const harmonious = enforceVowelHarmony(parsed);
				const validDipthongs = removeInvalidDipthongs(harmonious);
				const newRecords = [...records, { word, font, class: data.class, parsed, harmonious, validDipthongs }];
				setRecords(newRecords);
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
				<Performance word={word} prediction={removeInvalidDipthongs(enforceVowelHarmony(parsed))} />
			</div>
			<AverageDegreesOfError records={records} />
			<table>
				<thead>
					<tr>
						<th>Word</th>
						<th>Font</th>
						<th>Class</th>
						<th>parse()</th>
						<th>Degree of error</th>
						<th>enforceVowelHarmony()</th>
						<th>Degree of error</th>
						<th>removeInvalidDipthongs()</th>
						<th>Degree of error</th>
					</tr>
				</thead>
				<tbody>
					{records.map((record) => (
						<tr key={JSON.stringify(record)}>
							<td>{record.word}</td>
							<td>{record.font}</td>
							<td>{record.class}</td>
							<td>{record.parsed}</td>
							<td>{accuracy(record.word, record.parsed).toPrecision(2)}</td>
							<td>{record.harmonious}</td>
							<td>{accuracy(record.word, record.harmonious).toPrecision(2)}</td>
							<td>{record.validDipthongs}</td>
							<td>{accuracy(record.word, record.validDipthongs).toPrecision(2)}</td>
						</tr>
					))}
				</tbody>
			</table>
		</main>
	)
}
