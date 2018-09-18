#!/usr/bin/python3
# https://www.godaddy.com/garage/how-to-install-postgresql-on-ubuntu-14-04/
# https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-16-04
# https://help.ubuntu.com/community/PostgreSQL


import psycopg2
 
def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        # Update connection string information obtained from the portal
        host = "201.187.98.131"#"localhost"#"201.187.98.131"
        port = "5432"
        user = "postgres"
        dbname = "desa"
        password = "4321gp"#"K4k4l4o="#"4321gp"
        #sslmode = "require"

        # Construct connection string
        conn_string = "host={0} port={1} user={2} dbname={3} password={4}".format(host, port, user, dbname, password)
        conn = psycopg2.connect(conn_string)
        #conn=psycopg2.connect("host=201.187.98.131 port=5432 dbname=desa user=postgres password=4321gp")
 
        # create a cursor
        cur = conn.cursor()
        
        # execute a statement
        cur.execute('SELECT * FROM micampo.alimento')
        rows = cur.fetchall()
        print("\nRows: \n")
        for row in rows:
            print("   ", row)
        
        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('\nDatabase connection closed.')
 
 
if __name__ == '__main__':
    connect()
