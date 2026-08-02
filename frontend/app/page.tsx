'use client'

import axios from 'axios';

import Image from "next/image";
import { useState, useEffect } from 'react';

import Lexigraph from './slices';

const BACKEND = 'http://localhost:5000';
const ALPHABET = ['ᡠ','ᡬ','ᡴ','ᡦ','ᡤ','ᡮ','ᠮ','ᠰ','ᠵ',
                  'ᠨ','ᡳ','ᡟ','ᠸ','ᡧ','ᡵ','ᠪ','᠈','ᠶ',
                  '᠉','ᠩ','ᠠ','ᠴ','ᠯ','ᡝ','ᡷ','ᡰ','ᡥ',
                  'ᠺ','ᠣ','ᡭ','ᡱ','ᡨ','ᡯ','ᡩ','ᡶ','\'᠋',
									'ᡡ',' ','*']
const COLORS = ['var(--letter-color-1)','var(--letter-color-2)','var(--letter-color-3)',
                'var(--letter-color-4)','var(--letter-color-5)','var(--letter-color-6)',
                'var(--letter-color-7)','var(--letter-color-8)','var(--letter-color-9)',
								'var(--letter-color-10)','var(--letter-color-11)','var(--letter-color-12)',
								'var(--letter-color-13)','var(--letter-color-14)','var(--letter-color-15)',
								'var(--letter-color-16)','var(--letter-color-17)','var(--letter-color-18)',
								'var(--letter-color-19)','var(--letter-color-20)','var(--letter-color-21)',
								'var(--letter-color-22)','var(--letter-color-23)','var(--letter-color-24)',
								'var(--letter-color-25)','var(--letter-color-26)','var(--letter-color-27)',
								'var(--letter-color-28)','var(--letter-color-29)','var(--letter-color-30)',
								'var(--letter-color-31)','var(--letter-color-32)','var(--letter-color-33)',
								'var(--letter-color-34)','var(--letter-color-35)','var(--letter-color-36)',
								'var(--letter-color-37)','white','white']
function LoadButton({ setWord }) {
	const onClick = () => {
		axios.get(`${BACKEND}/corpus/random`)
			.then(({ data }) => {
				setWord(data.manchu);
			});
	};

	return <button type="button" onClick={onClick}>LOAD RANDOM WORD</button>;
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

function SaveButton({ word, font, boundaries, setResults }) {
	const onClick = () => {
		axios.put(`${BACKEND}/lexigraphy/save/${font}/${word}`, { boundaries })
			.then((response) => {
				setResults(response.data);
				alert('Save succesful');
			})
			.catch((error) => {
				console.error(error);
				alert('Save failed');
			});
	};
	return (
		<button
			type="submit"
			onClick={onClick}
		>
			SAVE LEXIGRAPH
		</button>
	)
}

function parseResults(results) {
	const characters = results
		.map(({ character }) => numberToCharacter(character));
	const reducedCharacters = [];
	reducedCharacters.push(characters[0]);

	characters.forEach((character) => {
		if (reducedCharacters[reducedCharacters.length - 1] == character) return;
		reducedCharacters.push(character);
	});

	return reducedCharacters.filter((character) => character !== '*').join('');
}

function numberToCharacter(number) {
	return ALPHABET[number];
}

function characterColor(characterNumber) {
	return COLORS[characterNumber];
}

function PredictionChartLegend({ results = PLACEHOLDER_RESULTS }) {
	const [uniqueCharacters, setUniqueCharacters] = useState([]);

	useEffect(() => {
		const newCharacters = [];
		results.forEach(({ character }) => {
			if (newCharacters.includes(character)) return;
			newCharacters.push(character);
		});
		setUniqueCharacters(newCharacters);
	}, [results]);

	return (
		<ul className="prediction-chart-legend">
			{uniqueCharacters.map((character) => (
				<li key={character}>
					<div style={{ background: characterColor(character) }} />
					{numberToCharacter(character)}
				</li>
			))}
		</ul>
	)
}

function OCRResults({ font, word, results = PLACEHOLDER_RESULTS }) {
	const url = `${BACKEND}/lexigraphy/new/${font}/${word}`;
	const parsed = parseResults(results);
	return (
		<div className="prediction">
			<img src={url} />
			{results.map(({ character, confidence }, index) => (
				<div
					className="slice-letter-prediction"
					style={{
						top: `${(index * 2) + 18}px`,
						width: `${20 + (confidence)}px`,
						background: characterColor(character),
					}}
				/>
			))}
			<PredictionChartLegend results={results} />
			<p className="manchu-text">[{parsed}]</p>
		</div>
	);
}

export default function Home() {
	const [word, setWord] = useState('ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ')
	const [boundaries, setBoundaries] = useState([])
	const [lexigraph, setLexigraph] = useState()
	const [font, setFont] = useState(0);
	const [results, setResults] = useState([]);

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
			<main className="labelling">
				<form onSubmit={(e) => { e.preventDefault(); }}>
					<div>
						<LoadButton setWord={setWord} />
						<FontSelection font={font} setFont={setFont} />
					</div>
					<div className="form-fields">
  					<BoundarySetter word={word} boundaries={boundaries} setBoundaries={setBoundaries} />
  					<Lexigraph word={word} font={font} boundaries={boundaries} size="large" />
  					<LexigraphWord word={word} font={font} />
					</div>
					<SaveButton word={word} font={font} boundaries={boundaries} setResults={setResults} />
				</form>
				<OCRResults font={font} word={word} results={results} />
      </main>
    </div>
  );
}
