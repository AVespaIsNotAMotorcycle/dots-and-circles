import levenshtein from 'js-levenshtein';

import styles from './Performance.module.css';

function accuracy(word, prediction) {
	const sumOfLengths = word.length + prediction.length;
	const distance = levenshtein(word, prediction);
	const degree = (sumOfLengths - distance) / sumOfLengths;
	return degree;
}

export default function Performance({ word, prediction }) {
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
