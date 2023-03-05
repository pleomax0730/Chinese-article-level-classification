import MySQLdb
from os import environ

CNX_product = {          
    'host': 'ponddychinese-production.cgjt0ap8pwdt.us-west-2.rds.amazonaws.com',
    'username': 'ponddy',
    'password': 'mWGrUVbmv5ya5gAz',
    'db': 'ebdb',
    'port': 3306}

CNX_dev = {
    #'host': 'ponddychinese-production.cgjt0ap8pwdt.us-west-2.rds.amazonaws.com',             
    'host': '',
    'username': 'ponddy',
    'password': 'mWGrUVbmv5ya5gAz',
    'db': 'engdb',
    'port': 3306}


CNX_local = {
    'host': '192.168.1.100',
    'username': 'owenzhong',
    'password': 'Ponddy2018',
    'db': 'mlebdb',
    'port': 3306}


## 本機mysql
#CNX = CNX_local
## ponddy chinese (dev)
#CNX = CNX_dev
## ponddy chinese (production)
#CNX = CNX_local
CNX = CNX_product

print(CNX)
class DB:
    conn = None

    def connect(self):
        print('host', CNX['host'])
        self.conn = MySQLdb.connect(CNX['host'], CNX['username'], CNX['password'], CNX['db'] , port=CNX['port'], charset='utf8')

    def query(self, sql):
        try:
            cursor = self.cursor()
            cursor.execute(sql)
        except (AttributeError, MySQLdb.OperationalError):
            self.connect()
            cursor = self.cursor()
            cursor.execute(sql)
        return cursor

    def IUD_execute(self, sql):
        try:
            cursor = self.cursor()
            cursor.execute(sql)
            self.conn.commit()
        except (AttributeError, MySQLdb.OperationalError):
            self.connect()
            cursor = self.cursor()
            cursor.execute(sql)
            self.conn.commit()
        return cursor
        
    def cursor(self):
        return self.conn.cursor(MySQLdb.cursors.DictCursor)

db = DB()
