'use client'

import axios from 'axios';

import styles from './slices.module.css';

import Image from "next/image";
import { useState, useEffect } from 'react';

const BACKEND = 'http://localhost:5000';

function Slice({ margin, length, size }) {
	const scale = size === 'large' ? 2 : 1;
	return (
		<div
			style={{
				marginTop: `${margin * scale}px`,
				height: `${(length) * scale}px`,
			}}
			className="slice"
		/>
	);
}

function Slices({ word, boundaries, size }) {
	if (boundaries.length !== word.length) return <div className="slices" />;
	return (
		<div className={styles.slices}>
			{word.split('').map((letter, index) => {
				return (
					<Slice
						size={size}
						margin={Number(boundaries[index][0])}
						length={Number(boundaries[index][1])}
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
			<Slices word={word} boundaries={boundaries} size={size} />
		</figure>
	);
}
