'use client'

import axios from 'axios';
import { useState, useEffect } from 'react';

import { numberToCharacter, characterColor } from '../alphabet';

import styles from './OCRVisualization.module.css';

const BACKEND = 'http://localhost:5000';
const PLACEHOLDER_RESULTS = [];

export function removeInvalidDipthongs(word) {
	const validDipthongs = ['ᠠᡳ',
													'ᠠᠣ',
                          'ᡝᡳ',
                          'ᡝᠣ',
                          'ᡳᠠ',
                          'ᡳᡝ',
                          'ᡳᡳ',
                          'ᡳᠣ',
                          'ᡳᡠ',
                          'ᠣᡳ',
                          'ᠣᠣ',
                          'ᡠᠠ',
                          'ᡠᡝ',
                          'ᡠᡳ',
                          'ᡠᠣ',
                          'ᡡᠠ',
                          'ᡡᡝ',
                          'ᡡᡳ',
                          'ᡡᠣ'];
	const vowels = ['ᡝ', 'ᠠ', 'ᠣ', 'ᡡ', 'ᡳ', 'ᡠ'];
	const validated = word.split('')
												.map((letter, index) => {
													if (index === 0) return letter;

													const previous = word[index - 1];
													if (!vowels.includes(letter)) return letter;
													if (!vowels.includes(previous)) return letter

													const dipthong = [previous, letter].join('');
													if (validDipthongs.includes(dipthong)) return letter;
													return undefined;
												})
												.filter((letter) => letter !== undefined)
												.join('');
	return validated;
} 

export function enforceVowelHarmony(word) {
	const frontVowels = ['ᡝ'];
	const backVowels = ['ᠠ', 'ᠣ', 'ᡡ'];
	const neutralVowels = ['ᡳ', 'ᡠ'];

	let frontScore = 0;
	let backScore = 0;
	word.split('').forEach((letter) => {
		if (frontVowels.includes(letter)) frontScore += 1;
		if (backVowels.includes(letter)) backScore += 1;
	});

	let harmoniousWord = word;
	if (frontScore > backScore) {
		harmoniousWord = harmoniousWord.replaceAll('ᠠ', 'ᡝ');
		harmoniousWord = harmoniousWord.replaceAll('ᠣ', 'ᡠ');
		harmoniousWord = harmoniousWord.replaceAll('ᡡ', 'ᡳ');
	}
	if (backScore > frontScore) {
		harmoniousWord = harmoniousWord
			.split('')
			.map((letter, index) => {
				if (letter === 'ᡝ' && harmoniousWord[index + 1] !== 'ᠣ') return 'ᠠ';
				return letter;
			})
			.join('');
	}
	return harmoniousWord;
}

export function parseResults(results) {
	const characters = results
		.map(({ character }) => numberToCharacter(character));
	const reducedCharacters = [];
	reducedCharacters.push();

	characters.forEach((character, index) => {
		if (reducedCharacters.length > 0 && reducedCharacters[reducedCharacters.length - 1] == character) return;
		if (characters[index + 1] != character || characters[index - 1] != character) return
		reducedCharacters.push(character);
	});

	const noBreaks = reducedCharacters.filter((character) => character !== '*').join('');
	return enforceVowelHarmony(noBreaks);
	// return removeInvalidDipthongs(enforceVowelHarmony(noBreaks));
  // return noBreaks;
}

export function PredictionChartLegend({ prediction = PLACEHOLDER_RESULTS }) {
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

export function RowPredictionVisualizer({ prediction, flip = false}) {
	return (
		<div className={styles.predictionVisualizer}>
			{prediction.map(({ character, confidence }, index) => (
				<div
					key={`${character}${confidence}${index}`}
					className={flip ? [styles.predictionVisualizerRow, styles.flip].join(' ') : styles.predictionVisualizerRow}
					style={{
						top: `${(index * 2) + 18}px`,
						width: `${10 + (confidence * 100)}px`,
						background: characterColor(character),
					}}
				/>
			))}
		</div>
	);
}

export default function OCRResults({ font, word, results = PLACEHOLDER_RESULTS, setParsed = () => {} }) {
	const [primaryPrediction, setPrimaryPrediction] = useState(results);
	const [secondaryPrediction, setSecondaryPrediction] = useState(results);

	const url = `${BACKEND}/lexigraphy/new/${font}/${word}`;
	const parsed1 = parseResults(primaryPrediction);
	const parsed2 = removeInvalidDipthongs(parseResults(secondaryPrediction));

	useEffect(() => {
		if (font == undefined) return;
		if (!word) return;
		axios.get(`${BACKEND}/lexigraphy/predict/${font}/${word}`)
			.then(({ data }) => {
				setPrimaryPrediction(data.primary_predictions);
				setSecondaryPrediction(data.secondary_predictions);
			});
	}, [font, word, results])

	useEffect(() => {
		setParsed(parsed2.trim());
	}, [parsed2]);

	return (
		<>
  		<div className="prediction">
  			<img src={url} />
  			<p
  				className="manchu-text"
  				style={{ marginTop: '20px', fontSize: '3.8rem', fontFamily: `manchu${font}` }}
  			>
  				{parsed1}
  			</p>
  			<div>
  				<RowPredictionVisualizer prediction={primaryPrediction} />
  			</div>
  		</div>
  		<div className="prediction">
  			<img src={url} />
  			<p
  				className="manchu-text"
  				style={{ marginTop: '20px', fontSize: '3.8rem', fontFamily: `manchu${font}` }}
  			>
  				{parsed2}
  			</p>
  			<div>
  				<RowPredictionVisualizer prediction={secondaryPrediction} />
  			</div>
  		</div>
		</>
	);
}
