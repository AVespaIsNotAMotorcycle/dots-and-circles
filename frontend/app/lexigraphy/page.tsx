'use client'

import styles from './lexigraphy.module.css';

import Link from "next/link";

import axios from 'axios';
import { useState, useEffect } from 'react';

const BACKEND = 'http://localhost:5000';

function formatFontName(fonts, fontNumber) {
	const fontName = fonts[Number(fontNumber)];
	return fontName.substring(9, fontName.length - 4);
}

function FontName({ font }) {
	const [fonts, setFonts] = useState();
	const url = `${BACKEND}/lexigraphy/fonts/dict`;

	useEffect(() => {
		axios.get(url)
			.then(({ data }) => { console.log(data); setFonts(data); })
			.catch(console.error);
	}, [font]);

	if (!fonts) return null;
	return <span>Font: {formatFontName(fonts, font)}</span>;
}

function Lexigraph({ font, manchu }) {
	const url = `${BACKEND}/lexigraphy/new/${font}/${manchu}`;
	return (
		<section className={styles.lexigraphCard}>
			<div>
				<div>
					<span>Abkai: {manchu}</span>
					<br />
					<FontName font={font} />
				</div>
  			<Link href={`/edit/${font}/${manchu}`}>
  				<button type="button">Edit</button>
  			</Link>
			</div>
			<img src={url} />
		</section>
	)
}

function Page({ start = 0, end = 12}) {
	const lexigraphs = [{ manchu: 'ᠰᡳᡰᡥᡳᡤᡳᠶᠠᠨ', font: 0 },
											{ manchu: 'ᠰᡳᡰᡥᡳᡤᡳᠶᠠᠨ', font: 0 },
											{ manchu: 'ᠰᡳᡰᡥᡳᡤᡳᠶᠠᠨ', font: 0 }];
	return (
		<div className={styles.lexigraphyPage}>
			{lexigraphs.map(({ font, manchu }) => (
				<Lexigraph key={`${font}${manchu}`} font={font} manchu={manchu} />
			))}
		</div>
	)
}

export default function Lexigraphy() {
	return (
		<main style={{ display: 'block' }}>
			<h1>Lexigraphy</h1>
			<Page />
		</main>
	);
}
