'use client'

import axios from 'axios';

import styles from './slices.module.css';

import Image from "next/image";
import { useState, useEffect } from 'react';

const BACKEND = 'http://localhost:5000';

function Slice({ margin, length }) {
	return (
		<div
			style={{
				marginTop: `${margin * 2}px`,
				height: `${(length) * 2}px`,
			}}
			className="slice"
		/>
	);
}

function Slices({ word, boundaries }) {
	if (boundaries.length !== word.length) return <div className="slices" />;
	return (
		<div className={styles.slices}>
			{word.split('').map((letter, index) => {
				return (
					<Slice
						margin={boundaries[index][0]}
						length={boundaries[index][1]}
						key={`${letter}-${index}`}
					/>
				);
			})}
		</div>
	);
}

export default function Lexigraph({ word, font, boundaries, size='normal' }) {
	const url = `${BACKEND}/lexigraphy/new/${font}/${word}`;

	const lexigraphClass = styles.lexigraph;
	const largeClass = styles.large;
	const normalClass = styles.normal;

	return (
		<figure
			className={size === 'large' ? `${lexigraphClass} ${largeClass}` : `${lexigraphClass} ${normalClass}`}
		>
			<figcaption className="element-label">Lexigraph</figcaption>
			<img src={url} />
			<Slices word={word} boundaries={boundaries} />
		</figure>
	);
}
