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
export function numberToCharacter(number) {
	return ALPHABET[number];
}

export function characterColor(characterNumber) {
	return COLORS[characterNumber];
}
