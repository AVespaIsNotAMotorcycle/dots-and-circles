from utils import to_abkai

ABKAI_PAIRS = [
               ['ᠰᡳᠨᡩᠠᡥᠠ', 'sindaha'],
               ['ᡤᡡᠰᠠᡳ', 'gvsai'],
               ['ᡤᠠᡳᡶᡳ', 'gaifi'],
               ['ᠪᡝᡳᠰᡝ', 'beise'],
               ['ᡥᡝᠩᡴᡳᠯᡝᡵᡝ', 'hengkilere'],
               ['ᠨᡳᠶᠠᠯᠮᠠ', 'niyalma'],
               ['ᡶᡝᠵᡳᠯᡝ', 'fejile'],
               ['ᡥᡡᠸᠠᠩᡩᡳ', 'hvwangdi'],
               ['ᡝᠮᡠ', 'emu'],
               ['ᠮᠠᠩᡤᡳ', 'manggi'],
               ['ᡨᡝᠨᡨᡝᡴᡝ', 'tenteke'],
               ['ᠠᠨᡳᠶᠠ ᠰᡝ', 'aniya se'],
               ['ᠶᠠᠮᠵᡳ', 'yamji'],
               ['ᠴᠣᠣᡥᠠᡳ ᡶᠠᡶᡠᠨ', 'qoohai fafun'],
               ['ᡨᡝᡴᠰᡳᠨ', 'teksin'],
               ['ᠰᠣᠩᡴᠣᡳ', 'songkoi'],
               ['ᡴᠠᡩᠠᠯᠠᡵᠠ', 'kadalara'],
               ['ᡷᡳ', 'jy'],
               ['ᠰᠠᠪᡠᡵᠠᡴᡡ ᠣᡥᠣ', 'saburakv oho'],
               ['ᡩᠣᠨᠵᡳᠴᡳ', 'donjiqi'],
               ['ᡨᡠᡴᡨᠠᠨ', 'tuktan'],
               # Didn't keep track of where these lines are from :(
               [
                   'ᠶᠠᠶᠠ ᠨᡳᠶᠠᠯᠮᠠ ᠪᡝ ᠈ ᠪᡳᡵᡝᠮᡝ ᡤᠣᠰᡳᠴᡳ ᠠᠴᠠᠮᠪᡳ ᠈ ᠠᠪᡴᠠ ᡤᡝᠮᡠ ᡝᠯᠪᡝᡥᡝᠪᡳ ᠈',
                   'yaya niyalma be , bireme gosiqi aqambi , abka gemu elbehebi ,',
               ],
               [
                   'ᠨᠠ ᡤᡝᠮᡠ ᠠᠯᡳᡥᠠᠪᡳ ᠈ ᠶᠠᠪᡠᠨ ᠸᡝᠰᡳᡥᡠᠨ ᠨᡳᠩᡤᡝ ᠈ ᡤᡝᠪᡠ ᡝᠰᡳ ᠠᠯᡤᡳᡴᠠ ᠈',
                   'na gemu alihabi , yabun wesihun ningge , gebu esi algika ,',
               ],
               [
                   'ᠨᡳᠶᠠᠯᠮᠠᡳ ᡠᠵᡝᠨ ᠣᠪᡠᡵᡝᠩᡤᡝ ᠈ ᠠᡵᠪᡠᠨ ᠸᡝᠰᡳᡥᡠᠨ ᠸᠠᡴᠠ ᠈ ᡝᡵᡩᡝᠮᡠ ᠠᠮᠪᠠ ᠨᡳᠩᡤᡝ ᠈',
                   'niyalmai ujen oburengge , arbun wesihun waka , erdemu amba ningge ,',
               ],
               [
                   'ᠠᠯᡤᡳᠨ ᡝᠰᡳ ᠪᠠᡩᠠᡵᠠᡴᠠ ᠈ ᠨᡳᠶᠠᠯᠮᠠᡳ ᡤᡡᠨᡳᠨ ᡩᠠᡥᠠᡵᠠᠩᡤᡝ ᠈ ᡤᡳᠰᡠᠨ ᡴᡡᠸᠠᠰᠠ ᠸᠠᡴᠠ ᠈',
                   'algin esi badaraka , niyalmai gvnin daharangge , gisun kvwasa waka ,',
               ],
               [
                   'ᠪᡝᠶᡝ ᡩᡝ ᠮᡠᡨᡝᠨ ᠪᡳᠴᡳ ᠈ ᡠᠮᡝ ᡝᠮᡥᡠᠨ ᠴᡳᠰᡠᠯᠠᡵᠠ ᠈ ᠨᡳᠶᠠᠯᠮᠠ ᡩᡝ ᠮᡠᡨᡝᠨ ᠪᡳᠴᡳ ᠈',
                   'beye de muten biqi , ume emhun qisulara , niyalma de muten biqi ,',
               ],
               # Profound Mirror of Classical Prose - Intro to the Profound Mirror of Classical Prose p 3
               [
                   'ᠠᡴᡡ ᠉ ᠯᡝᡠᠯᡝᠮᡝ ᡤᡳᠰᡠᡵᡝᡵᡝ ᡥᠠᠴᡳᠨ ᡩᡝ ᠣᠴᡳ᠉ ᠰᡠᠮᡝ ᡥᠠᡶᡠᡴᡳᠶᠠᡵᠠ ᠪᡝ',
                   'akv . leoleme gisurere haqin de oqi. sume hafukiyara be',
               ],
               [
                   'ᡩᠠ ᠣᠪᡠᡥᠠᠪᡳ ᠈ ᡝᡵᡝ ᡳ ᡤᡳᠩ ᠴᡳ ᡩᡝᡵᡳᠪᡠᡥᡝᠩᡤᡝ ᠉ ᡨᡠᠴᡳᠪᡠᠮᡝ',
                   'da obuhabi , ere i ging qi deribuhengge . tuqibume',
               ],
               [
                   'ᠸᡝᠰᡳᠮᠪᡠᡵᡝ ᡥᠠᠴᡳᠨ ᡩᡝ ᠣᠴᡳ ᠈ ᠪᠠᡩᠠᡵᠠᠮᠪᡠᠮᡝ ᠨᡝᡳᠯᡝᡵᡝ ᠪᡝ ᠵᡠᡵᡤᠠᠨ',
                   'wesimbure haqin de oqi , badarambume neilere be jurgan',
               ],
               [
                   'ᠣᠪᡠᡥᠠᠪᡳ ᠈ ᡝᡵᡝ ᡧᡠ ᡤᡳᠩ ᠴᡳ ᡩᡝᡵᡳᠪᡠᡥᡝᠩᡤᡝ ᠉ ᡶᡠ ᠈ ᠰᡠᠩ ᠨᡳ',
                   'obuhabi , ere xu ging qi deribuhengge . fu , sung ni',
               ],
               [
                   'ᡥᠠᠴᡳᠨ ᡩᡝ ᠣᠴᡳ ᡩᠠᡵᡳᠮᡝ ᡠᠯᡥᡳᠪᡠᡵᡝ ᠪᡝ ᠵᠣᡵᡳᠨ ᠣᠪᡠᡥᠠᠪᡳ ᠉ ᡝᡵᡝ',
                   'haqin de oqi darime ulhibure be jorin obuhabi . ere',
               ],
               [
                   'ᡧᡳ ᡤᡳᠩ ᠴᡳ ᡩᡝᡵᡳᠪᡠᡥᡝᠩᡤᡝ ᠉ ᠵᡠᠸᠠᠨ ᠈ ᠰᡳᡠᡳ ᡳ ᡥᠠᠴᡝᡳᠨ ᡩᡝ',
                   'xi ging qi deribuhengge . juwan , sioi i haqein de',
               ],
               # Tales of the 120 Old Men - Debtelin 06 Tale 02 p 1
               [
                   'ᡝᠮᡠ ᠰᠠᡴᡩᠠ ᡥᡝᠨᡩᡠᠮᡝ ᡩᡝᡴᡩᡝᠨᡳ ᡤᡳᠰᡠᠨ ᡨᠣᠨᡩᠣ ᠠᠮᠪᠠᠨ ᠪᡝ',
                   'emu sakda hendume dekdeni gisun tondo amban be',
               ],
               [
                   'ᡥᡳᠶᠣᡠᡧᡠᠩᡤᠠ ᠵᡠᡳ ᠴᡳ ᠪᠠᡳᠰᡠ ᠰᡝᡥᡝᠪᡝ ᡩᠣᠨᠵᡳᡥᠠᡴᡡᠨ ᡤᡠᠴᡠᠰᡝ ᡩᡝ',
                   'hiyooxungga jui qi baisu sehebe donjihakvn guquse de',
               ],
               [
                   'ᠪᡳ ᠠᠯᠠᡵᠠ ᡥᡳᠶᠣᡠᡧᡠᠨ ᠰᡝᡵᡝᠩᡤᡝ ᠨᡳᠶᠠᠯᠮᠠᡳ ᡩᠠ ᠪᠠᠨᡳᠨ ᠨᡳᠶᠠᠯᠮᠠ',
                   'bi alara hiyooxun serengge niyalmai da banin niyalma',
               ],
               [
                   'ᡨᠣᠮᡝ ᡩᠠᠴᡳ ᠪᡳᠰᡳᡵᡝᠩᡤᡝ ᡥᡳᠶᠣᡠᡧᡠᠨ ᠠᡴᡡ ᠣᠴᡳ ᡩᠠ ᠪᠠᠨᡳᠨ',
                   'tome daqi bisirengge hiyooxun akv oqi da banin',
               ],
               [
                   'ᠪᡠᡵᡠᠪᡠᡶᡳ ᡩᠠ ᡤᡡᠨᡳᠨ ᠰᠠᠰᠠ ᠪᡠᡵᡠᠪᡠᡵᡝ ᠪᡝ ᡩᠠᡥᠠᠮᡝ ᡥᠣᠯᠣ ᠴᠠᠩᡤᡳ',
                   'burubufi da gvnin sasa burubure be dahame holo qanggi',
               ],
               [
                   'ᠣᡶᡳ ᡨᠣᠨᡩᠣ ᠠᠮᠪᠠᠨ ᠣᠮᡝ ᠮᡠᡨᡝᠮᠪᡳᡠ ᠠᠮᠠ ᡝᠮᡝ ᠮᡠᠰᡝ ᠪᡝ',
                   'ofi tondo amban ome mutembio ama eme muse be',
               ],
               [
                   'ᠪᠠᠨᠵᡳᡶᡳ ᠠᠵᡳᡤᠠᠨ ᠴᡳ ᡨᠠᠩᠰᡠᠯᠠᠮᡝ ᡤᠣᠰᡳᡥᠠᡳ ᠮᡠᡨᡠᡨᡝᠯᡝ ᡳᠰᡳᠨᠠᡵᠠᡴᡡ ᠪᠠ',
                   'banjifi ajigan qi tangsulame gosihai mututele isinarakv ba',
               ],
               [
                   'ᠠᡴᡡ ᡤᠣᠰᡳᠮᡝ ᡠᠵᡳᠮᡝ ᡨᠠᠴᡳᡥᡳᠶᠠᠮᡝ ᡥᡡᠸᠠᡧᠠᠪᡠᡥᠠᡳ ᠰᠠᡴᡩᠠ ᠪᡝᠶᡝ ᠪᡝᡩᡝᡵᡝᡨᡝᠯᡝ',
                   'akv gosime ujime taqihiyame hvwaxabuhai sakda beye bederetele',
               ],
               [
                   'ᡤᡡᠨᡳᠨ ᠠᠨ ᡳ ᠸᠠᠵᡳᡵᠠᡴᡡ ᠠᡳᡴᠠ ᠰᠠᡵᡴᡡ ᠰᡝᠴᡳ ᡩᠠᠮᡠ ᠮᡠᠰᡝᡳ',
                   'gvnin an i wajirakv aika sarkv seqi damu musei',
               ],
              ]

def test_to_abkai():
    for manchu, abkai in ABKAI_PAIRS:
        assert abkai == to_abkai(manchu)
