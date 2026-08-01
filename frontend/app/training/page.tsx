'use client'

import axios from 'axios';
import { useState, useEffect} from 'react';

import styles from './training.module.css';

const URL = 'http://localhost:5000/train';

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
				{training && `Training... epoch ${results.length} / ${epochs}`}
				<table>
					<thead>
						<th>Epoch</th>
						<th>Accuracy</th>
					</thead>
					<tbody>
						{results.map(({ accuracy }, index) => (
							<tr>
								<td>{index + 1}</td>
								<td>{accuracy}</td>
							</tr>
						))}
					</tbody>
				</table>
			</section>
		</main>
	);
}
