import argparse

# type1
def get_typ1_kw(kw, db='etl_refresh_prod', table=None, source='exp'):
    """
    Args:
        texts (List[str]): The list of texts to classify.

    Returns:
        List[List[Tuple[str, float]]]: A list of classification results for each text,
        where each result is a list of tuples containing the predicted label and the
        confidence score.
    """
    kw_list = kw.split(',')
    kw_list = [x.lower().strip() for x in kw_list if x.strip() != ""]

    main_kw = kw_list[0]

    aux_kw_count = len(kw_list[1:])
    final_aux_sring = 'or'
    kw_count = 1

    if (not table) and (source == 'exp'):
        table = 'ln_flattened_rawdata'
        typ2_base_sample_query = f'''T2.website,T1.exp.member_id ,T1.exp.date_from,exp.date_to, T1.exp.description
        FROM {db}.{table} as T1
        join {db}."lnk2domain_mapping" as T2
        ON T1.exp.company_url = T2.ln_company
        '''
    elif (not table) and (source == 'job'):
        table = 'linkedin_jobs_coresignal_rawdata'
        typ2_base_sample_query = f'''SELECT T2.website , created ,last_updated, lower(description) as description
        FROM {db}.{table} as T1
        join {db}."lnk2domain_mapping" as T2
        ON T1.company_url = T2.ln_company
        '''
    else:
        print("ERROR: Please select the write source i.e\n 'exp' : for linkden experience \n 'job' : for Jobs data  ")
        return ("ERROR: Please select the write source i.e 'exp' : for linkden experience \n 'job' : for Jobs data  ")

    main_kw = f'''where description like '% {main_kw.strip()} %' '''

    for aux in kw_list[1:]:

        if kw_count < aux_kw_count:
            search_txt = f''' description like '% {aux.strip()} %' '''

            final_aux_sring = final_aux_sring + search_txt + 'OR'

        elif kw_count == aux_kw_count:
            search_txt = f''' description like '% {aux.strip()} %' '''
            final_aux_sring = final_aux_sring + search_txt

        kw_count = kw_count + 1

    if len(kw_list) > 1:
        query_2_sample = typ2_base_sample_query + main_kw + final_aux_sring

    if len(kw_list) == 1:
        query_2_sample = typ2_base_sample_query + main_kw

    print(query_2_sample)


def get_typ2_kw(kw, db='etl_refresh_prod', table=None, source='exp'):
    """
    A function to generate a sample query for type 2 keyword search based on given parameters.
    """
    kw_list = kw.split(',')
    kw_list = [x.lower().strip() for x in kw_list if x.strip() != ""]

    if len(kw_list) < 2:
        print('Kw is type 2 , it needs atleast 2 keywords.')
        return ('Kw is type 2 , it needs atleast 2 keywords.')

    if len(kw_list) == 2:

        if (not table) and (source == 'exp'):
            table = 'ln_flattened_rawdata'
            typ2_base_sample_query = f'''SELECT T2.website,T1.exp.member_id ,T1.exp.date_from,exp.date_to, T1.exp.description
            FROM {db}.{table} as T1
            join {db}."lnk2domain_mapping" as T2
            ON T1.exp.company_url = T2.ln_company
            where T1.exp.description like '% {kw_list[0].strip()} %' and T1.exp.description like '% {kw_list[1].strip()} %' 
            '''
            print(typ2_base_sample_query)
            return

        elif (not table) and (source == 'job'):
            table = 'linkedin_jobs_coresignal_rawdata'
            typ2_base_sample_query = f'''SELECT T2.website , created ,last_updated, lower(description) as description
            FROM {db}.{table} as T1
            join {db}."lnk2domain_mapping" as T2
            ON T1.company_url = T2.ln_company
            where t1.description like '% {kw_list[0].strip()} %' and t1.description like '% {kw_list[1].strip()} %'
            '''
            print(typ2_base_sample_query)
            return
        else:
            print("ERROR: Please select the write source i.e 'exp' : for linkden experience \n 'job' : for Jobs data  ")
            return (
                "ERROR: Please select the write source i.e 'exp' : for linkden experience \n 'job' : for Jobs data  ")

    main_kw = kw_list[0]
    aux_kw_count = len(kw_list[1:])
    final_aux_sring = 'and ('
    kw_count = 1

    if (not table) and (source == 'exp'):
        table = 'ln_flattened_rawdata'
        typ2_base_sample_query = f'''SELECT T2.website,T1.exp.member_id ,T1.exp.date_from,exp.date_to, T1.exp.description as des 
        FROM etl_refresh_prod.ln_flattened_rawdata as T1
        join etl_refresh_prod."lnk2domain_mapping" as T2
        ON T1.exp.company_url = T2.ln_company
            '''
        main_kw = f'''where T1.exp.description like '% {main_kw.strip()} %' '''

        for aux in kw_list[1:]:

            if kw_count < aux_kw_count:
                search_txt = f''' T1.exp.description like '% {aux.strip()} %' '''

                final_aux_sring = final_aux_sring + search_txt + 'OR'

            elif kw_count == aux_kw_count:
                search_txt = f''' T1.exp.description like '% {aux.strip()} %' '''
                final_aux_sring = final_aux_sring + search_txt + ')'

            kw_count = kw_count + 1

        query_2_sample = typ2_base_sample_query + main_kw + final_aux_sring
        print(query_2_sample)
        return query_2_sample

    elif (not table) and (source == 'job'):

        table = 'linkedin_jobs_coresignal_rawdata'
        typ2_base_sample_query = f'''SELECT T2.website , created ,last_updated, lower(description) as description
        FROM {db}.{table} as T1
        join {db}."lnk2domain_mapping" as T2
        ON T1.company_url = T2.ln_company
        '''
    else:
        print("ERROR: Please select the write source i.e\n 'exp' : for linkden experience \n 'job' : for Jobs data  ")
        return ("ERROR: Please select the write source i.e\n 'exp' : for linkden experience \n 'job' : for Jobs data  ")

    main_kw = f'''where description like '% {main_kw.strip()} %' '''

    for aux in kw_list[1:]:

        if kw_count < aux_kw_count:
            search_txt = f''' description like '% {aux.strip()} %' '''

            final_aux_sring = final_aux_sring + search_txt + 'OR'

        elif kw_count == aux_kw_count:
            search_txt = f''' description like '% {aux.strip()} %' '''
            final_aux_sring = final_aux_sring + search_txt + ')'

        kw_count = kw_count + 1

    query_2_sample = typ2_base_sample_query + main_kw + final_aux_sring

    print(query_2_sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the keywords and source")
    parser.add_argument("--kw_type", type=str, help="The type of keyword", required=True)
    parser.add_argument("--kw", type=str, help="The keyword to search for", required=True)
    parser.add_argument("--db", type=str, help="The database to search in", required=False, default="etl_refresh_prod")
    parser.add_argument("--table", type=str, help="The table to search in", required=False, default=None)
    parser.add_argument("--source", type=str, help="The source of the data", required=True)
    args = parser.parse_args()

    if args.kw_type == "1":
        get_typ1_kw(args.kw, args.db, args.table, args.source)
    if args.kw_type == "2":
        get_typ2_kw(args.kw, args.db, args.table, args.source)