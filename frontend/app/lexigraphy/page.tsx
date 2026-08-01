'use client'

import styles from './lexigraphy.module.css';

import Link from "next/link";

import axios from 'axios';
import { useState, useEffect } from 'react';

import Lexigraph from '../slices';

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
			.then(({ data }) => { setFonts(data); })
			.catch(console.error);
	}, [font]);

	if (!fonts) return null;
	return <span>Font: {formatFontName(fonts, font)}</span>;
}

function LexigraphCard({ font, manchu, boundaries }) {
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
			<Lexigraph font={font} word={manchu} boundaries={boundaries} />
		</section>
	)
}

function Page({ start = 0, end = 12, lexigraphs = []}) {
	return (
		<div className={styles.lexigraphyPage}>
			{lexigraphs.map(([manchu, font, boundaries]) => (
				<LexigraphCard key={`${font}${manchu}`} font={font} manchu={manchu} boundaries={boundaries} />
			))}
		</div>
	)
}

const PER_PAGE = 12;
function Pagination({ range, setRange }) {
	const go = (delta) => {
		let start = range.start + delta;
		let end = range.end + delta;
		if (start < 0) { start = 0; end = PER_PAGE; }
		setRange({ start, end });
	}
	const goBack = () => { go(PER_PAGE * -1) };
	const goForward = () => { go(PER_PAGE) };

	return (
		<div>
			<button type="button" onClick={goBack}>Previous</button>
			{`Showing lexigraphs ${range.start} through ${range.end}`}
			<button type="button" onClick={goForward}>Next</button>
		</div>
	)
}

export default function Lexigraphy() {
	const [range, setRange] = useState({ start: 0, end: PER_PAGE });
	const [page, setPage] = useState([]);

	useEffect(() => {
		axios.get(`${BACKEND}/lexigraphy/get/page?start=${range.start}&end=${range.end}`)
			.then(({ data }) => { setPage(data); })
			.catch(console.error);
	}, [range]);

	return (
		<main style={{ display: 'block' }}>
			<h1>Lexigraphy</h1>
			<Pagination range={range} setRange={setRange} />
			<Page start={range.start} end={range.end} lexigraphs={page} />
		</main>
	);
}
