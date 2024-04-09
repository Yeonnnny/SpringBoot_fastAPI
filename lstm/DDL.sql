-- 상품 (product)

DROP TABLE product;

CREATE TABLE product
(
    product_id VARCHAR2(50) PRIMARY KEY  
    , product_name VARCHAR2(200) NOT NULL 
    , product_desc CLOB NOT NULL
    , product_img VARCHAR2(2000) NOT NULL
    , price NUMBER NOT NULL
    , origin VARCHAR2(30) NOT NULL
    , moq NUMBER NOT NULL
    , unit VARCHAR2(10) NOT NULL
    , category VARCHAR2(20) NOT NULL check(category in ('FOOD_BEVERAGE','CHEMICAL','HEALTH_MEDICAL','ELECTRONIC','COSMETIC'))
    , create_date date DEFAULT sysdate
    , update_date date
    , remote_ip VARCHAR2(20) NOT NULL
    , country VARCHAR2(50)
    , hit_count NUMBER DEFAULT 0
    , lstm_predict_proba NUMBER(5,2) NOT NULL
    , lstm_predict CHAR(1)  NOT NULL check(lstm_predict in (0,1))
    , judge CHAR(1) check(judge in ('N','Y'))
    , customer_id VARCHAR2(20) NOT NULL
    , product_delete CHAR(1) DEFAULT 'N' check(product_delete in ('N','Y'))
);
select * from product;


-- 첨부 파일에 대한 컬럼 추가
alter table product add original_file_name varchar2(200);
alter table product add saved_file_name varchar2(200);

alter table product drop column product_img;
