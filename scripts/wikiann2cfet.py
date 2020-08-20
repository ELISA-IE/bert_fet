"""
This script extracts fine-grained entity typing data from the wikiann data in a
MongoDB database.

Usage:
python wikiann2cfet.py \
    --entity_type_file /shared/nas/data/m1/panx2/data/KBs/dump/dbpedia/dbpedia_2016-04/core/output/yago_types_wordnet.json \
    --ontology /shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/types.rare.txt \
    --output /shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/en/en.cfet.json \
    --port 27017
"""

import json
import random
import logging
from collections import Counter
from argparse import ArgumentParser, Namespace

from pymongo import MongoClient


logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_data(entity_type_file: str,
                 ontology_file: str,
                 output_file: str,
                 title_field_name: str = 'title',
                 db_port: int = 27017):
    """Extracts data from the database.

    Args:
        entity_type_file (str): path to the entity type file. This file is in
            JSON format, where the keys are entity titles and values are
            corresponding lists of entity types.
        ontology_file (str): path to the ontology file. This file is in TXT
            format, where each line is a target entity type.
        output_file (str): path to the output file.
        title_field_name (str, optional): Field name for entity titles. Defaults
            to 'title'.
        db_port (int): MongoDB database port. Defaults to 27017.
    
    Notes:

    Entity type file example:
    ```
    {
        "!!!":[
            "Abstraction100002137",
            "DanceBand108249960",
            "SocialGroup107950920",
            "YagoLegalActor",
            "Organization108008335",
            "YagoPermanentlyLocatedEntity",
            "MusicalOrganization108246613",
            "YagoLegalActorGeo",
            "Group100031264",
            "RockGroup108250501"
        ],
        "!!!_(album)":[
            "Whole100003553",
            "Medium106254669",
            "Album106591815",
            "PhysicalEntity100001930",
            "Instrumentality103575240",
            "Object100002684",
            "Artifact100021939"
        ],
    }

    Ontology file example:

    ```
    WorldOrganization108294696
    Disease114070360
    Symptom114299637
    MedicalPractitioner110305802
    Drone102207179
    IllHealth114052046
    Region108630985
    House103544360
    Club108227214
    Round104113641
    ```
    """
    # Load the mapping table from entity titles to type lists from file
    logger.info('Loading entity types from {}'.format(entity_type_file))
    entity_type_map = json.load(open(entity_type_file))
    logger.info('#Entities: {}'.format(len(entity_type_map)))

    # Load target types from the ontology file
    logger.info('Loading target entity types from {}'.format(ontology_file))
    type_set = set([t.strip() for t in open(ontology_file).read().split('\n')
                if t.strip()])
    logger.info('#Target Entity Types: {}'.format(len(type_set)))

    doc_mention_count = Counter()
    valid_num = entity_num = not_matched_num = title_num = 0

    with open(output_file, 'w', encoding='utf-8') as w:
        with MongoClient(host='127.0.0.1', port=db_port) as client:
            col = client['enwiki']['sentences']

            for doc_idx, doc in enumerate(col.find({'len_links': {'$gt': 0}}), 1):
                doc_id = doc['id']
                links = doc['links']
                entities = []
                for link in links:
                    if title_field_name not in link:
                        continue
                    title = link[title_field_name]
                    title = title.replace(' ', '_')
                    if title in entity_type_map:
                        title_num += 1
                        # Get types using entity title
                        types = entity_type_map[title]
                        # Only keep target types
                        types = [t for t in types if t in type_set]
                        # If the entity has at least target types
                        if types:
                            entities.append((link['text'],
                                             link['start'],
                                             link['end'],
                                             types))

                # If the sentence contains valid entities
                if entities:
                    entity_num += 1
                    tokens = doc['tokens']
                    # Map from offsets to indices
                    token_start_offset_map = {}
                    token_end_offset_map = {}
                    for token_idx, token in enumerate(tokens):
                        token_start_offset_map[token['start']] = token_idx
                        token_end_offset_map[token['end']] = token_idx + 1
                    
                    annotations = []
                    for idx, (text, start, end, types) in enumerate(entities):
                        if (start not in token_start_offset_map
                            or end not in token_end_offset_map):
                            not_matched_num += 1
                            continue
                        start = token_start_offset_map[start]
                        end = token_end_offset_map[end]
                        annotations.append({
                            'mention': text,
                            'mention_id': '{}-{}'.format(
                                doc_id, doc_mention_count[doc_id]),
                            'start': start,
                            'end': end,
                            'labels': types
                        })
                        doc_mention_count[doc_id] += 1

                    # If the sentence has any annotations
                    if annotations:
                        valid_num += 1
                        w.write(json.dumps({
                            'tokens': [t['text'] for t in tokens],
                            'annotations': annotations
                        }) + '\n')
                
                if doc_idx % 1000 == 0:
                    print('\r#Processed: {}, #Entities: {}, #Valid: {}, #NotMatched: {}, #Titles: {}'.format(
                        doc_idx, entity_num, valid_num, not_matched_num, title_num),
                        end='')


def parse_arguments() -> Namespace:
    """Parses commandline arguments.

    Returns:
        Namespace: a namespace object for parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-e', '--entity_type_file',
                        help='Path to the entity type mapping file')
    parser.add_argument('-n', '--ontology_file',
                        help='Path to the ontology file')
    parser.add_argument('-o', '--output',
                        help='Path to the output file')
    parser.add_argument('-p', '--port', type=int, default=27017,
                        help='MongoDB database port (default = 27017).')
    parser.add_argument('-t', '--title', default='title',
                        help='Field name for entity titles.')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    extract_data(args.entity_type_file,
                 args.ontology_file,
                 args.output,
                 args.title,
                 args.port)


if __name__ == '__main__':
    main()