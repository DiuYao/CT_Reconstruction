//#if !defined( __INITFILE_H )
//#define __INITFILE_H
#pragma once
#ifdef _WIN32
#ifdef Soft_IMPLEMENT_INIT 
#define Soft_API /*extern "C"*/ __declspec(dllexport) 
#else 
#define Soft_API /*extern "C" */__declspec(dllimport) 
#endif 
#else
#define Soft_API
#endif
#include <map>  
#include <string>  
#include <vector>
using namespace std; 

#if 0 //_WIN32
#include "afxwin.h"
#define String CString
#define LPSTR1 LPCSTR
#define MGetPrivateProfileString(a,b,c,d,e,f) GetPrivateProfileString(a,b,c,d,e,f)
#define MWritePrivateProfileString(a,b,c,d) WritePrivateProfileString(a,b,c,d)
#else
#define String string
#ifndef _WIN32
#define LPSTR1 const char*
#define BOOL int
// #define __int64 long long
typedef long long  __int64;
#else
#define LPSTR1 const char*
typedef int                 BOOL;
#endif
#define MGetPrivateProfileString(a,b,c,d,e,f) GetKey(a,b,c,d)
#define MWritePrivateProfileString(a,b,c,d) SetKey(a,b,c)
#endif
#define CONFIGLEN 256
//every line length must less than CONFIGLEN
class Soft_API  CInitFile
{
public:
	CInitFile();
	~CInitFile();

private:	
	char	m_buffer[CONFIGLEN];
	String	m_szModName;
typedef struct{string key,values,comment;} KeyVC;
//              子键索引    子键值   
typedef map<string,int> KEYMAP;  
typedef struct { KEYMAP keymap; vector<KeyVC> mkey; } MapKeyVC;
//                    主键索引               主键值    
//typedef map<string,KEYMAP> MAINKEYMAP;  
	// MAINKEYMAP m_Map;
typedef struct { string sessionname, comment; MapKeyVC msession; } SMapKeyVC;
KEYMAP sessionmap;
vector<SMapKeyVC> msessionkey;
int GetKey(const char* mAttr, const char* cAttr, const char* Defaultv, char* value);
int SetKey(const char* mAttr, const char* cAttr, String value);
void ParseFile();
bool haveSave, haveChange;

public:	
	void	GetFileName(LPSTR1 szFileName = "Reconstruct.txt" );
	void	SaveFile(LPSTR1 szFileName = 0);
	String  GetParaFilePath(){return m_szModName;}
	BOOL	SetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, String szValue);
	BOOL	SetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, char iValue);
	BOOL	SetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, short iValue);
	BOOL	SetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, int iValue);
	BOOL	SetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, unsigned long iValue);
	BOOL	SetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, double dbValue);
	BOOL	SetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, __int64 i64value);
	BOOL	SetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, size_t iValue);
	BOOL	SetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, float iValue);
	

	BOOL	GetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, LPSTR1 szDefault, String& szValue);
	BOOL	GetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, int iDefault, char& iValue);
	BOOL	GetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, int iDefault, short& iValue);
	BOOL	GetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, int iDefault, int& iValue);
	BOOL    GetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, long iDefault, long& iValue);
	BOOL	GetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, double dbDefault, double& dbValue);
	BOOL    GetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, __int64 i64Default, __int64& i64Value);
	BOOL    GetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, size_t iDefault, size_t& iValue);
	BOOL    GetEntryValue(LPSTR1 szSection, LPSTR1 szEntry, float iDefault, float& iValue);
	
	KEYMAP&GetSessionMap(){return sessionmap;}
	vector<SMapKeyVC>& GetSessionKeyValues(){return msessionkey;}
	#ifdef WithDecode
	void	SaveFileTxt(int code,LPSTR1 szFileName = 0);
	void	DecodeFileTxt(int code,LPSTR1 szFileName);
	#endif
};
#if _WIN32
#ifndef Soft_IMPLEMENT_INIT
#ifdef _DEBUG
#pragma comment(lib, "InitFileD.lib")
#else
#pragma comment(lib, "InitFile.lib")
#endif
#endif
#endif
