'use client'

import axios from 'axios';

import Image from "next/image";
import { useState, useEffect } from 'react';

const BACKEND = 'http://localhost:5000';

function LoadButton({ setWord }) {
	const onClick = () => {
		axios.get(`${BACKEND}/corpus/random`)
			.then(({ data }) => {
				setWord(data.manchu);
			});
	};

	return <button type="button" onClick={onClick}>LOAD RANDOM WORD</button>;
}

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
	console.log(boundaries);
	if (boundaries.length !== word.length) return <div className="slices" />;
	return (
		<div className="slices">
			{word.split('').map((letter, index) => {
				console.log(boundaries[index]);
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

function LexigraphWord({ word, font }) {
	const fontName = `manchu${font}`

	return (
		<section className="lexigraph-word">
			<span className="element-label">Text</span>
  		<span className="manchu-text" style={{ fontFamily: fontName }}>
      	{word.split('').map((letter, index) => (
      		<span key={`${letter}-${index}`}>{letter}</span>
      	))}
  		</span>
		</section>
	);
}

function FontSelection({ font, setFont }) {
	const [fonts, setFonts] = useState({});

	useEffect(() => {
		axios.get(`${BACKEND}/lexigraphy/fonts/dict`)
			.then(({ data }) => { setFonts(data); });
	}, [])

	const onChange = ({ target }) => { setFont(target.value); };

	return (
		<select value={font} onChange={onChange}>
			{Object.keys(fonts).map((key) => <option value={key} key={key}>{fonts[key]}</option>)}
		</select>
	);
}

function BoundarySetter({word, boundaries, setBoundaries}) {
	const setMargin = (index, value) => {
		if (value < 1) return;
		if (index < 0 || index > word.length) return;
		const newBoundaries = [...boundaries];
		newBoundaries[index][0] = value;
		setBoundaries(newBoundaries);
	}
	const setLength = (index, value) => {
		if (value < 1) return;
		const newBoundaries = [...boundaries];
		newBoundaries[index][1] = value;
		setBoundaries(newBoundaries);
	}

	if (boundaries.length !== word.length) return;
	return (
		<section className="boundary-setter">
			{word.split('').map((letter, index) => (
				<div key={`${letter}${index}`}>
					<span className="letter manchu-text">{letter}</span>
					<label>
						Margin:
						<input
							type="number"
							value={boundaries[index][0]}
							onChange={({ target }) => { setMargin(index, target.value); }}
						/>
					</label>
					<label>
						Length:
						<input
							type="number"
							value={boundaries[index][1]}
							onChange={({ target }) => { setLength(index, target.value); }}
						/>
					</label>
				</div>
			))}
		</section>
	);
}

function Lexigraph({ word, font, boundaries }) {
	const url = `${BACKEND}/lexigraphy/new/${font}/${word}`;
	return (
		<figure>
			<figcaption className="element-label">Lexigraph</figcaption>
			<img src={url} />
			<Slices word={word} boundaries={boundaries} />
		</figure>
	);
}

export default function Home() {
	const [word, setWord] = useState('ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ')
	const [boundaries, setBoundaries] = useState([])
	const [lexigraph, setLexigraph] = useState()
	const [font, setFont] = useState(0);

	useEffect(() => {
		const newBoundaries = [];
		const offset = 10;
		const spacing = 10;
		const gap = 1;
		word.split('').forEach((letter, index) => {
			const margin = index == 0 ? offset : gap;
			const length = spacing;
			newBoundaries.push([margin, length]);
		})
		setBoundaries(newBoundaries);
	}, [word])

  return (
    <div className="flex flex-col flex-1 items-center justify-center bg-zinc-50 font-sans dark:bg-black">
			<main>
			  <LoadButton setWord={setWord} />
				<FontSelection font={font} setFont={setFont} />
				<form>
					<BoundarySetter word={word} boundaries={boundaries} setBoundaries={setBoundaries} />
					<Lexigraph word={word} font={font} boundaries={boundaries} />
					<LexigraphWord word={word} font={font} />
				</form>
      </main>
    </div>
  );
}
