'use client'

import axios from 'axios';
import { useState, useEffect } from 'react';

import { numberToCharacter, characterColor } from './alphabet';

const BACKEND = 'http://localhost:5000';
const PLACEHOLDER_RESULTS = [];

function parseResults(results) {
	const characters = results
		.map(({ character }) => numberToCharacter(character));
	const reducedCharacters = [];
	reducedCharacters.push();

	characters.forEach((character, index) => {
		if (reducedCharacters.length > 0 && reducedCharacters[reducedCharacters.length - 1] == character) return;
		if (characters[index + 1] != character || characters[index - 1] != character) return
		reducedCharacters.push(character);
	});

	return reducedCharacters.filter((character) => character !== '*').join('');
}

function PredictionChartLegend({ prediction = PLACEHOLDER_RESULTS }) {
	const [uniqueCharacters, setUniqueCharacters] = useState([]);

	useEffect(() => {
		const newCharacters = [];
		prediction.forEach(({ character }) => {
			if (newCharacters.includes(character)) return;
			newCharacters.push(character);
		});
		setUniqueCharacters(newCharacters);
	}, [prediction]);

	return (
		<ul className="prediction-chart-legend">
			{uniqueCharacters.map((character) => (
				<li key={character} style={{ fontSize: '1.8rem' }}>
					<div style={{ background: characterColor(character), margin: '0.5rem' }} />
					{numberToCharacter(character)}
				</li>
			))}
		</ul>
	)
}

function RowPredictionVisualizer({ prediction }) {
	return (
		<>
			{prediction.map(({ character, confidence }, index) => (
				<div
					className="slice-letter-prediction"
					style={{
						top: `${(index * 2) + 18}px`,
						width: `${10 + (confidence * 100)}px`,
						background: characterColor(character),
					}}
				/>
			))}
			<PredictionChartLegend prediction={prediction} />
		</>
	);
}

export default function OCRResults({ font, word, results = PLACEHOLDER_RESULTS, setParsed = () => {} }) {
	const [prediction, setPrediction] = useState(results);

	const url = `${BACKEND}/lexigraphy/new/${font}/${word}`;
	const parsed = parseResults(prediction);

	useEffect(() => {
		if (font == undefined) return;
		if (!word) return;
		axios.get(`${BACKEND}/lexigraphy/predict/${font}/${word}`)
			.then(({ data }) => { setPrediction(data.predictions); });
	}, [font, word, results])

	useEffect(() => {
		setParsed(parsed);
	}, [parsed]);

	return (
		<div className="prediction">
			<img src={url} />
			<p
				className="manchu-text"
				style={{ marginTop: '20px', fontSize: '3.8rem', fontFamily: `manchu${font}` }}
			>
				{parsed}
			</p>
			<RowPredictionVisualizer prediction={prediction} />
		</div>
	);
}
