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

function Slice({ start, end }) {
	return <div style={{ height: `${(end - start) * 2}px`}} className="slice" />;
}

function Slices({ word, boundaries }) {
	return (
		<div className="slices">
			{word.split('').map((letter, index) => (
				<Slice
					start={index === 0 ? 0 : boundaries[index - 1]}
					end={boundaries[index]}
					key={`${letter}-${index}`}
				/>
			))}
		</div>
	);
}

function Caption({ word, font }) {
	const fontName = `manchu${font}`

	return (
		<figcaption className="manchu-text" style={{ fontFamily: fontName }}>
    	{word.split('').map((letter, index) => (
    		<span key={`${letter}-${index}`}>{letter}</span>
    	))}
		</figcaption>
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

function Lexigraph({ word, font, boundaries }) {
	const url = `${BACKEND}/lexigraphy/new/${font}/${word}`;
	return (
		<>
			<img src={url} />
			<Slices word={word} boundaries={boundaries} />
		</>
	);
}

export default function Home() {
	const [word, setWord] = useState('ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ')
	const [sliceBoundaries, setSliceBoundaries] = useState([])
	const [lexigraph, setLexigraph] = useState()
	const [font, setFont] = useState(0);

	useEffect(() => {
		const boundaries = [15,23,31,37,46,54,65,73,86,103];
		/*
		const boundaries = [];
		word.split().forEach((letter, index) => { boundaries.push(index + 1); })
		*/
		setSliceBoundaries(boundaries);
	}, [])

  return (
    <div className="flex flex-col flex-1 items-center justify-center bg-zinc-50 font-sans dark:bg-black">
			<main>
			  <LoadButton setWord={setWord} />
				<FontSelection font={font} setFont={setFont} />
				<figure>
					<Lexigraph word={word} font={font} boundaries={sliceBoundaries} />
					<Caption word={word} font={font} />
				</figure>
      </main>
    </div>
  );
}
