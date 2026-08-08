import styles from './LexigraphClassDescription.module.css';

export default function LexigraphClassDescription({ lexigraphClass }) {
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
