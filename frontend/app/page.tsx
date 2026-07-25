'use client'

import Image from "next/image";
import { useState, useEffect } from 'react';

function Slice({ start, end }) {
	return <div style={{ height: `${end - start * 5}px`}} className="slice" />;
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

export default function Home() {
	const [word] = useState('ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ')
	const [sliceBoundaries, setSliceBoundaries] = useState([])

	useEffect(() => {
		const boundaries = [];
		word.split().forEach((letter, index) => { boundaries.push(index + 1); })
		setSliceBoundaries(boundaries);
	}, [])

  return (
    <div className="flex flex-col flex-1 items-center justify-center bg-zinc-50 font-sans dark:bg-black">
			<main>
				<button type="button">LOAD RANDOM WORD</button>
				<figure>
					<img src="tmpbun5si6r.PNG" />
					<Slices word={word} boundaries={sliceBoundaries} />
					<figcaption className="manchu-text">{word}</figcaption>
				</figure>
      </main>
    </div>
  );
}
