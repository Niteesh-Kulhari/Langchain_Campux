from langchain_classic.text_splitter import CharacterTextSplitter

text = """
Moon exploration is one of humanity's greatest scientific adventures. From the first footsteps of Apollo astronauts to modern robotic missions, 

we continue to study the Moon’s surface, rocks, water ice, and environment. Future missions aim to build lunar bases, test new technologies, and use the Moon as a stepping stone toward Mars. 

Exploring the Moon helps us understand our past and prepare for humanity’s future in space.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

chunks = splitter.split_text(text)

for chunk in chunks:
    print(chunk)
    print("\n")