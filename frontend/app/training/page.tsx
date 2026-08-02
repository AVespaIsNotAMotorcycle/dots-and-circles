'use client'

import axios from 'axios';
import { useState, useEffect} from 'react';

import styles from './training.module.css';
import { numberToCharacter, characterColor } from '../alphabet';

const URL = 'http://localhost:5000/train';

function toPercent(number) {
	return Number(number / 100).toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 2 });
}

function EpochTable({ results }) {
	return (
		<table>
			<thead>
				<th>Epoch</th>
				<th>Accuracy</th>
			</thead>
			<tbody>
				{results.map(({ accuracy }, index) => (
					<tr>
						<td>{index + 1}</td>
						<td>{toPercent(accuracy)}</td>
					</tr>
				))}
			</tbody>
		</table>
	);
}

function LetterTable({ results }) {
	const letters = {};

	if (!results[0]) return;
	if (!results[0].trials) return;

	const tabulate = ({ actual, correct }) => {
		if (!letters[actual]) letters[actual] = {};
		if (!letters[actual].attempts) letters[actual].attempts = 0;
		if (!letters[actual].successes) letters[actual].successes = 0;

		letters[actual].attempts += 1;
		if (correct) letters[actual].successes += 1;
	};

	results.forEach(({ trials }) => {
		trials.forEach(tabulate);
	})

	return (
		<table>
			<thead>
				<th>Letter</th>
				<th>Trials</th>
				<th>Accuracy</th>
			</thead>
			<tbody>
				{Object.keys(letters).map((letter) => {
  				const accuracy = letters[letter].successes / letters[letter].attempts * 100;
					return (
  					<tr key={letter}>
  						<td>{numberToCharacter(letter)}</td>
  						<td>{letters[letter].attempts}</td>
  						<td>{toPercent(accuracy)}</td>
  					</tr>
					);
				})}
			</tbody>
		</table>
	);
}

function TrainingProgressTracker({ epochs, results, training}) {
	return (
		<div className={styles.progressTracker}>
			{training && `Training... epoch ${results.length} / ${epochs}`}
			{!training && results.length === epochs && 'Done!'}
		</div>
	);
}

export default function Training() {
	const [epochs, setEpochs] = useState(1); 
	const [training, setTraining] = useState(false);
	const [results, setResults] = useState([]);

	const addToResults = (result) => {
		const newResults = [...results, result];
		setResults(newResults);
	};
	useEffect(() => {
		if (!training) return;

		axios.put('http://localhost:5000/train')
			.then(({ data }) => {
				if (results.length == epochs - 1) setTraining(false);
				addToResults(data);
			});

	}, [training, results])

	const onSubmit = async (event) => {
		event.preventDefault();
		setTraining(true);
		setResults([]);
	};

	return (
		<main>
			<h1>Training</h1>
			<form className={styles.settings} onSubmit={onSubmit}>
				<label>
					Number of epochs:
					<input type="number" value={epochs} onChange={({ target }) => { setEpochs(target.value); }} />
				</label>
				<button type="submit">Train</button>
			</form>
			<section>
				<h2>Results</h2>
				<TrainingProgressTracker epochs={epochs} results={results} training={training} />
				<div className={styles.tables}>
					<EpochTable results={results} />
					<LetterTable results={results} />
				</div>
			</section>
		</main>
	);
}
